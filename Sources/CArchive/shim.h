/* Stable public libarchive reader ABI, supplied by macOS. SDK distributions
   omit its headers, so declare only the functions used by the verifier. */
#include <sys/types.h>
#include <sys/stat.h>
#include <stdint.h>
#include <stddef.h>
struct archive;
struct archive_entry;
#define ARCHIVE_OK 0
#define ARCHIVE_EOF 1
#define AE_IFREG S_IFREG
#define AE_IFDIR S_IFDIR
#define AE_IFLNK S_IFLNK
struct archive *archive_read_new(void);
int archive_read_free(struct archive *);
int archive_read_support_filter_all(struct archive *);
int archive_read_support_format_tar(struct archive *);
int archive_read_open_filename(struct archive *, const char *, size_t);
int archive_read_next_header(struct archive *, struct archive_entry **);
ssize_t archive_read_data(struct archive *, void *, size_t);
const char *archive_entry_pathname(struct archive_entry *);
const char *archive_entry_hardlink(struct archive_entry *);
const char *archive_entry_symlink(struct archive_entry *);
mode_t archive_entry_filetype(struct archive_entry *);
int64_t archive_entry_size(struct archive_entry *);
int archive_read_set_options(struct archive *, const char *);
