# okc: problem=muse_decode_layer
"""Dynamic Muse decoder layer: fused boundaries, split-K small projections,
losslessly repacked NVFP4 single-token experts, and Marlin batched experts.
No activation-derived state is retained between calls.
"""
import types
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
HIDDEN=8192
RMS_EPS=1e-5
POST_NORM_EPS=1e-8
FP8=torch.float8_e4m3fn
N_EXPERTS=256
TOP_K=8
LOCAL_EXPERTS=32
D_EXPERT=4096
BLOCK_SIZE_M_CANDIDATES=(8,16,32,48,64)


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _marlin_experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


@triton.jit
def _qk_store(X, Q, KC, VC, Pos, Slots, KS, VS, ROPE:tl.constexpr):
    t=tl.program_id(0); h=tl.program_id(1); d=tl.arange(0,64)
    off=tl.where(h<16,h*64,1024+(h-16)*64)
    z=tl.load(X+t*2304+off+d).to(tl.float32)
    z=z*tl.rsqrt(tl.sum(z*z,0)/64+1.e-5)
    if ROPE:
        other=tl.load(X+t*2304+off+(d^1)).to(tl.float32)
        # RMS is shared by each even/odd pair.
        raw=tl.load(X+t*2304+off+d).to(tl.float32)
        inv=tl.rsqrt(tl.sum(raw*raw,0)/64+1.e-5)
        other=other*inv
        angle=tl.load(Pos+t).to(tl.float32)*tl.exp((-tl.log(500000.0))*(d//2*2).to(tl.float32)/64)
        z=z*libdevice.cos(angle)+tl.where(d%2==0,-other,other)*libdevice.sin(angle)
    if h<16:
        tl.store(Q+t*1024+h*64+d,z)
    else:
        slot=tl.load(Slots+t); hh=h-16
        tl.store(KC+slot*128+hh*64+d,z/tl.load(KS))
        v=tl.load(X+t*2304+1152+hh*64+d).to(tl.float32)
        tl.store(VC+slot*128+hh*64+d,v/tl.load(VS))


@triton.jit
def _out_gate(A,X,O):
    t=tl.program_id(0); h=tl.program_id(1); d=tl.arange(0,64)
    a=tl.load(A+t*1024+h*64+d).to(tl.float32)
    g=tl.load(X+t*2304+1280+h*64+d).to(tl.float32)
    a=a*tl.rsqrt(tl.sum(a*a,0)/64+1.e-5)/(1+tl.exp(-g))
    tl.store(O+t*1024+h*64+d,a)


@triton.jit
def _boundary(A,S,H,HF,SO,Alpha,Beta,W, FIRST:tl.constexpr, SLICED:tl.constexpr):
    t=tl.program_id(0); d=tl.arange(0,8192)
    if SLICED:
        a=tl.load(A+t*1024+d,d<1024,0).to(tl.float32)
    else:
        a=tl.load(A+t*8192+d).to(tl.float32)
    if not FIRST:
        a=a*(1+tl.load(W+d).to(tl.float32))
    a=a*tl.rsqrt(tl.sum(a*a,0)/8192+1.e-8)
    s=tl.load(S+t*8192+d)*tl.load(Alpha+d)+a*tl.load(Beta+d)
    tl.store(SO+t*8192+d,s)
    h=s*tl.rsqrt(tl.sum(s*s,0)/8192+1.e-5)
    # W points at the pre-FFN weight on the first boundary, and the
    # post-FFN weight on the second; its next norm is passed separately.
    if FIRST:
        h=h*(1+tl.load(W+d).to(tl.float32))
        tl.store(HF+t*8192+d,h)
    tl.store(H+t*8192+d,h)


@triton.jit
def _finish(A,S,H,SO,Alpha,Beta,Post,Next,SLICED:tl.constexpr):
    t=tl.program_id(0); d=tl.arange(0,8192)
    if SLICED:
        a=tl.load(A+t*1024+d,d<1024,0).to(tl.float32)
    else:
        a=tl.load(A+t*8192+d).to(tl.float32)
    a=a*(1+tl.load(Post+d).to(tl.float32))
    a=a*tl.rsqrt(tl.sum(a*a,0)/8192+1.e-8)
    s=tl.load(S+t*8192+d)*tl.load(Alpha+d)+a*tl.load(Beta+d)
    tl.store(SO+t*8192+d,s)
    h=s*tl.rsqrt(tl.sum(s*s,0)/8192+1.e-5)*(1+tl.load(Next+d).to(tl.float32))
    tl.store(H+t*8192+d,h)


@triton.jit
def _pre_norm(A,W,O):
    t=tl.program_id(0); d=tl.arange(0,4096)
    a=tl.load(A+t*4096+d).to(tl.float32)
    a=a*tl.rsqrt(tl.sum(a*a,0)/4096+1.e-5)*(1+tl.load(W+d).to(tl.float32))
    tl.store(O+t*4096+d,a)


@triton.jit
def _routing(L,B,Start,IDs,Local,Weights):
    t=tl.program_id(0); e=tl.arange(0,256); k=tl.arange(0,8)
    scores=1/(1+tl.exp(-tl.load(L+t*256+e)*0.1767766952966369))
    v=scores+tl.load(B+e)
    outids=tl.full((8,),0,tl.int32); outw=tl.full((8,),0,tl.float32)
    for i in range(8):
        idx=tl.argmax(v,0)
        val=tl.sum(tl.where(e==idx,scores,0),0)
        outids=tl.where(k==i,idx,outids); outw=tl.where(k==i,val,outw)
        v=tl.where(e==idx,-float('inf'),v)
    tl.store(IDs+t*8+k,outids)
    tl.store(Local+t*8+k,(outids-tl.load(Start)+256)%256)
    tl.store(Weights+t*8+k,outw/(tl.sum(outw,0)+1.e-15))


@triton.jit
def _dense_split(X,W,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,NS:tl.constexpr, BN:tl.constexpr=64,BK:tl.constexpr=64):
    n=tl.program_id(0)*BN+tl.arange(0,BN); sp=tl.program_id(1)
    m=tl.arange(0,16); kk=sp*(K//NS)+tl.arange(0,BK)
    acc=tl.full((16,BN),0,tl.float32)
    for i in range(K//NS//BK):
        a=tl.load(X+m[:,None]*K+kk[None,:],m[:,None]<M,0)
        b=tl.load(W+n[None,:]*K+kk[:,None],n[None,:]<N,0)
        acc+=tl.dot(a,b)
        kk+=BK
    tl.store(P+sp*M*N+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))


@triton.jit
def _dense_sum(P,O,M:tl.constexpr,N:tl.constexpr,NS:tl.constexpr,B:tl.constexpr=256):
    i=tl.program_id(0)*B+tl.arange(0,B); s=tl.arange(0,NS)
    a=tl.load(P+s[:,None]*M*N+i[None,:],i[None,:]<M*N,0)
    tl.store(O+i,tl.sum(a,0),i<M*N)


def _fast_gemm(x,w):
    m,k=x.shape; n=w.shape[0]
    if m<=8:
        ns=32 if k>=4096 else 8
        p=torch.empty((ns,m,n),device=x.device,dtype=torch.float32)
        out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
        _dense_split[(triton.cdiv(n,64),ns)](x,w,p,m,n,k,ns,num_warps=4)
        _dense_sum[(triton.cdiv(m*n,256),)](p,out,m,n,ns,num_warps=4)
        return out
    return _gemm(x,w)


def prepare(weights):
    router=weights['router']; hi=router.to(torch.bfloat16)
    state={'dense':{n:weights[n].contiguous() for n in ('qkv_gate','o','pooler','unpooler')},
           'weights':weights,'router':router.contiguous(),
           'router_hi_t':hi.T.contiguous(),'router_lo_t':(router-hi.float()).to(torch.bfloat16).T.contiguous(),
           'bank':_marlin_bank(weights),'post_expert_w':(1+weights['post_expert_norm'].float()).to(torch.bfloat16),
           'side_stream':torch.cuda.Stream(weights['qkv_gate'].device)}
    for key in ('w13','w2'):
        state[key+'_packed_t']=weights[key].transpose(1,2).contiguous()
        state[key+'_sf_t']=weights[key+'_sf'].transpose(1,2).contiguous()
    for prefix,name in [('a','attn_residual_gate'),('f','ffn_residual_gate')]:
        z=weights[name].float()/0.3; b=torch.sigmoid(z)
        state[prefix+'alpha']=torch.sqrt((torch.sigmoid(-z)*(1+b)).clamp(min=1.e-3))
        state[prefix+'beta']=b
    return state


def kernel_fn(data,state):
    cfg=data['config']; t=cfg['tokens']; b=cfg['batch']; length=cfg['q_len']; dev=data['attn_input'].device
    w=state['weights']; dense=state['dense']
    qkv=_fast_gemm(data['attn_input'],dense['qkv_gate'])
    q=torch.empty((t,16,64),device=dev,dtype=FP8)
    _qk_store[(t,18)](qkv,q,data['k_cache'],data['v_cache'],data['positions'],data['cache_loc'],data['k_scale'],data['v_scale'],bool(cfg['rope_theta']),num_warps=4)
    descale={}
    if cfg['kv_scale']!=1.0:
        descale=dict(k_descale=data['k_scale'].reshape(1,1).expand(b,2),v_descale=data['v_scale'].reshape(1,1).expand(b,2))
    out=flash_attn_with_kvcache(q=q,k_cache=data['k_cache'],v_cache=data['v_cache'],page_table=data['page_table'],cache_seqlens=data['cache_seqlens'],cu_seqlens_q=data['cu_seqlens_q'],max_seqlen_q=length,softmax_scale=cfg['sm_scale']*cfg['qk_scale_factor'],causal=True,window_size=(cfg['window']-1,0) if cfg['window'] else (-1,-1),scheduler_metadata=data['scheduler_metadata'],**descale)
    if isinstance(out,tuple): out=out[0]
    gated=torch.empty((t,1024),device=dev,dtype=torch.bfloat16)
    _out_gate[(t,16)](out,qkv,gated,num_warps=4)
    a=_fast_gemm(gated,dense['o'])
    h=torch.empty((t,8192),device=dev,dtype=torch.bfloat16); hf=torch.empty((t,8192),device=dev,dtype=torch.float32); s=torch.empty_like(data['stream'])
    _boundary[(t,)](a,data['stream'],h,hf,s,state['aalpha'],state['abeta'],w['pre_ffn_norm'],True,False,num_warps=8)
    main=torch.cuda.current_stream(); side=state['side_stream']; side.wait_stream(main)
    p=_fast_gemm(h,dense['pooler'])
    ids=torch.empty((t,8),device=dev,dtype=torch.int32); local=torch.empty_like(ids); tw=torch.empty((t,8),device=dev,dtype=torch.float32)
    with torch.cuda.stream(side):
        if t>8:
            logits=torch.mm(h,state['router_hi_t'],out_dtype=torch.float32)
            logits+=torch.mm(h,state['router_lo_t'],out_dtype=torch.float32)
        else: logits=F.linear(hf,state['router'])
        _routing[(t,)](logits,w['router_bias'],data['local_start'],ids,local,tw,num_warps=4)
    main.wait_stream(side)
    if not torch.cuda.is_current_stream_capturing():
        for z in (ids,local,tw): z.record_stream(main)
    pn=torch.empty_like(p); _pre_norm[(t,)](p,w['pre_expert_norm'],pn,num_warps=4)
    y=_experts(pn,tw,local,state)
    u=_fast_gemm(y,dense['unpooler'][:1024] if t<=8 else dense['unpooler'])
    hidden=torch.empty_like(h); so=torch.empty_like(s)
    _finish[(t,)](u,s,hidden,so,state['falpha'],state['fbeta'],w['post_ffn_norm'],w['next_pre_attn_norm'],t<=8,num_warps=8)
    return hidden,so,ids


@triton.jit
def _compact_routes(IDs, Rows, Counts, T:tl.constexpr, R:tl.constexpr):
    e=tl.program_id(0); i=tl.arange(0,R)
    match=(i<T*8)&(tl.load(IDs+i,i<T*8,255)==e)
    rank=tl.cumsum(match.to(tl.int32),0)-1
    tl.store(Rows+e*T+rank,i,match)
    tl.store(Counts+e,tl.sum(match.to(tl.int32),0))


@triton.jit
def _route_schedule(Counts, Ex, Starts, Total, T:tl.constexpr, BM:tl.constexpr, B:tl.constexpr):
    e=tl.program_id(0); j=tl.arange(0,32)
    counts=tl.load(Counts+j); blocks=tl.cdiv(counts,BM)
    base=tl.sum(tl.where(j<e,blocks,0),0)
    count=tl.load(Counts+e); q=tl.arange(0,B)
    tl.store(Ex+base+q,e,q<tl.cdiv(count,BM))
    tl.store(Starts+base+q,q*BM,q<tl.cdiv(count,BM))
    if e==0: tl.store(Total,tl.sum(blocks,0))


@triton.jit
def _fp4(code):
    mag=code&7
    f=((mag.to(tl.int32)<<22)+0x3f000000).to(tl.float32,bitcast=True)
    f=tl.where(mag<2,mag.to(tl.float32)*0.5,f)
    return tl.where((code&8)!=0,-f,f)


@triton.jit
def _packed_moe(X,W,SF,GS,Rows,Counts,Ex,Starts,Total,P,
                T:tl.constexpr,N:tl.constexpr,NS:tl.constexpr,STAGE:tl.constexpr,
                BM:tl.constexpr=16,BN:tl.constexpr=128,BK:tl.constexpr=128):
    nb=tl.program_id(0); block=tl.program_id(1); sp=tl.program_id(2)
    if block<tl.load(Total):
        e=tl.load(Ex+block); start=tl.load(Starts+block)
        mr=start+tl.arange(0,BM); count=tl.load(Counts+e)
        slot=tl.load(Rows+e*T+mr,mr<count,0)
        if STAGE==1: xr=slot//8
        else: xr=slot
        nn=nb*BN+tl.arange(0,BN)
        ki=sp*(4096//NS)
        acc=tl.full((BM,BN),0,tl.float32)
        for it in range(4096//NS//BK):
            kk=ki+tl.arange(0,BK)
            a=tl.load(X+xr[:,None]*4096+kk[None,:],mr[:,None]<count,0)
            kp=ki//2+tl.arange(0,BK//2)
            packed=tl.load(W+e*2048*N+kp[:,None]*N+nn[None,:]).to(tl.int32)
            pair=tl.arange(0,2)
            codes=(packed[:,None,:]>>(pair[None,:,None]*4))&15
            vals=_fp4(codes).reshape(BK,BN)
            sk=ki//16+tl.arange(0,BK//16)
            sf=tl.load(SF+e*256*N+sk[:,None]*N+nn[None,:]).to(tl.float32)
            vals=(vals.reshape(BK//16,16,BN)*sf[:,None,:]).reshape(BK,BN).to(tl.bfloat16)
            acc+=tl.dot(a,vals)
            ki+=BK
        acc*=tl.load(GS+e)
        tl.store(P+sp*T*8*N+slot[:,None]*N+nn[None,:],acc,mr[:,None]<count)


@triton.jit
def _packed_silu(P,IDs,O,T:tl.constexpr,NS:tl.constexpr,B:tl.constexpr=256):
    slot=tl.program_id(0); d=tl.program_id(1)*B+tl.arange(0,B)
    if tl.load(IDs+slot)<32:
        sp=tl.arange(0,NS)
        g=tl.sum(tl.load(P+sp[:,None]*T*8*8192+slot*8192+d[None,:]),0).to(tl.bfloat16).to(tl.float32)
        u=tl.sum(tl.load(P+sp[:,None]*T*8*8192+slot*8192+4096+d[None,:]),0).to(tl.bfloat16).to(tl.float32)
        tl.store(O+slot*4096+d,g/(1+tl.exp(-g))*u)


@triton.jit
def _packed_norm(P,IDs,TW,W,O,T:tl.constexpr,NS:tl.constexpr):
    t=tl.program_id(0); d=tl.arange(0,4096)
    total=tl.full((4096,),0,tl.float32)
    gain=1+tl.load(W+d).to(tl.float32)
    for k in range(8):
        slot=t*8+k
        if tl.load(IDs+slot)<32:
            y=tl.full((4096,),0,tl.float32)
            for sp in range(NS):
                y+=tl.load(P+sp*T*8*4096+slot*4096+d)
            y=y.to(tl.bfloat16).to(tl.float32)*gain
            y=y.to(tl.bfloat16).to(tl.float32)
            y=y*tl.rsqrt(tl.sum(y*y,0)/4096+1.e-8)
            total+=y*tl.load(TW+slot)
    tl.store(O+t*4096+d,total)


def _packed_experts(x,tw,ids,state):
    t=x.shape[0]; dev=x.device; w=state['weights']; bm=16
    rows=torch.empty((32,t),device=dev,dtype=torch.int32)
    counts=torch.empty((32,),device=dev,dtype=torch.int32)
    maxblocks=min(t*8,triton.cdiv(t*8,bm)+32)
    ex=torch.empty((maxblocks,),device=dev,dtype=torch.int32)
    starts=torch.empty_like(ex); total=torch.empty((),device=dev,dtype=torch.int32)
    _compact_routes[(32,)](ids,rows,counts,t,triton.next_power_of_2(t*8),num_warps=4)
    _route_schedule[(32,)](counts,ex,starts,total,t,bm,triton.next_power_of_2(triton.cdiv(t,bm)),num_warps=4)
    ns=4 if t<=32 else 1
    p1=torch.empty((ns,t*8,8192),device=dev,dtype=torch.float32)
    inter=torch.empty((t*8,4096),device=dev,dtype=torch.bfloat16)
    p2=torch.empty((ns,t*8,4096),device=dev,dtype=torch.float32)
    _packed_moe[(64,maxblocks,ns)](x,state['w13_packed_t'],state['w13_sf_t'],w['w13_gs'],rows,counts,ex,starts,total,p1,t,8192,ns,1,num_warps=4)
    _packed_silu[(t*8,16)](p1,ids,inter,t,ns,num_warps=4)
    _packed_moe[(32,maxblocks,ns)](inter,state['w2_packed_t'],state['w2_sf_t'],w['w2_gs'],rows,counts,ex,starts,total,p2,t,4096,ns,2,num_warps=4)
    out=torch.empty((t,4096),device=dev,dtype=torch.bfloat16)
    _packed_norm[(t,)](p2,ids,tw,w['post_expert_norm'],out,t,ns,num_warps=8)
    return out


def _experts(x,topk_weights,topk_ids,state):
    # Shape-only dispatch: the packed path was faster on single-token decode.
    if x.shape[0] == 1:
        return _packed_experts(x,topk_weights,topk_ids,state)
    return _marlin_experts(x,topk_weights,topk_ids,state)

