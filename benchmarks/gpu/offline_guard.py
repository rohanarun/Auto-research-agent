"""Linux seccomp: deny IPv4/IPv6 sockets, including localhost, before importing a candidate."""
import ctypes,errno,platform,socket

def install(sync=False):
 class Filter(ctypes.Structure):_fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint)]
 class Program(ctypes.Structure):_fields_=[('len',ctypes.c_ushort),('filter',ctypes.POINTER(Filter))]
 arch=platform.machine();nr={'x86_64':41,'aarch64':198}[arch]
 # Load syscall nr; all syscalls except socket pass. Check socket domain.
 audit={'x86_64':0xc000003e,'aarch64':0xc00000b7}[arch]
 ops=[(0x20,0,0,4),(0x15,1,0,audit),(0x06,0,0,0x80000000),(0x20,0,0,0),(0x54,0,0,0x3fffffff),*(op for number in [425,426,427] for op in [(0x15,0,1,number),(0x06,0,0,0x00050000|errno.EPERM)]),(0x15,0,3,nr),(0x20,0,0,16),(0x15,2,0,socket.AF_INET),(0x15,1,0,socket.AF_INET6),(0x06,0,0,0x7fff0000),(0x06,0,0,0x00050000|errno.EPERM)]
 array=(Filter*len(ops))(*(Filter(*x) for x in ops));program=Program(len(ops),array);libc=ctypes.CDLL(None,use_errno=True)
 if libc.prctl(38,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'no_new_privs failed')
 result=libc.syscall({'x86_64':317,'aarch64':277}[arch],1,1,ctypes.byref(program)) if sync else libc.prctl(22,2,ctypes.byref(program),0,0)
 if result!=0:raise OSError(ctypes.get_errno(),'seccomp installation failed')
 for domain in [socket.AF_INET,socket.AF_INET6]:
  try:socket.socket(domain,socket.SOCK_STREAM)
  except PermissionError:pass
  else:raise RuntimeError('Network guard failed')
