"""Unprivileged Linux filesystem confinement for one benchmark job."""
import ctypes,os,platform,stat

EXECUTE=1;WRITE_FILE=2;READ_FILE=4;READ_DIR=8
REMOVE_DIR=16;REMOVE_FILE=32;MAKE_CHAR=64;MAKE_DIR=128;MAKE_REG=256;MAKE_SOCK=512;MAKE_FIFO=1024;MAKE_BLOCK=2048;MAKE_SYM=4096;REFER=8192;TRUNCATE=16384

def install(read_roots,write_roots,sync=False):
 if platform.system()!='Linux':raise RuntimeError('Landlock requires Linux')
 libc=ctypes.CDLL(None,use_errno=True)
 abi=libc.syscall(444,ctypes.c_void_p(0),0,1)
 if abi<1:raise OSError(ctypes.get_errno(),'Landlock unavailable; refusing unconfined job')
 if sync and abi<8:raise RuntimeError('Thread-synchronized Landlock ABI8 required')
 handled=(1<<13)-1
 if abi>=2:handled|=REFER
 if abi>=3:handled|=TRUNCATE
 class Ruleset(ctypes.Structure):_fields_=[('handled_access_fs',ctypes.c_uint64),('handled_access_net',ctypes.c_uint64),('scoped',ctypes.c_uint64)]
 class PathBeneath(ctypes.Structure):
  _pack_=1
  _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
 attr=Ruleset(handled,3 if abi>=4 else 0,2 if abi>=6 else 0);size=24 if abi>=6 else (16 if abi>=4 else 8);rs=libc.syscall(444,ctypes.byref(attr),size,0)
 if rs<0:raise OSError(ctypes.get_errno(),'Landlock ruleset creation')
 try:
  for root,write in [(p,False) for p in read_roots]+[(p,True) for p in write_roots]:
   root=os.path.realpath(root)
   if not os.path.exists(root):continue
   fd=os.open(root,os.O_PATH|os.O_CLOEXEC)
   try:
    directory=stat.S_ISDIR(os.fstat(fd).st_mode)
    rights=handled if write and directory else (EXECUTE|WRITE_FILE|READ_FILE if write else EXECUTE|READ_FILE)
    if directory:rights|=READ_DIR
    rights&=handled
    rule=PathBeneath(rights,fd)
    if libc.syscall(445,rs,1,ctypes.byref(rule),0)<0:raise OSError(ctypes.get_errno(),'Landlock path '+root)
   finally:os.close(fd)
  if libc.prctl(38,1,0,0,0)!=0 or libc.syscall(446,rs,8 if sync else 0)<0:raise OSError(ctypes.get_errno(),'Landlock enforcement')
 finally:os.close(rs)
 return abi
