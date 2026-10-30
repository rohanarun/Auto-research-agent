"""Read untrusted worker outputs without following links across job/UID boundaries."""
import os,stat

def read_job_file(directory,name,owners,limit,tail=None):
 parent=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  descriptor=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
  try:
   info=os.fstat(descriptor)
   if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_uid not in owners or info.st_size>limit:raise PermissionError('Unsafe worker output ownership, type, link count or size')
   with os.fdopen(descriptor,'rb',closefd=False) as stream:
    if tail is not None:stream.seek(max(0,info.st_size-tail))
    return stream.read(limit+1 if tail is None else tail)
  finally:os.close(descriptor)
 finally:os.close(parent)
