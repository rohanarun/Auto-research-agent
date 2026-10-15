import Foundation
import CryptoKit
import CArchive

public enum DaytonaArchive {
    public static func sha256(_ file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom:file);defer { try? handle.close() };var hash = SHA256()
        while let data = try handle.read(upToCount:1_048_576),!data.isEmpty { hash.update(data:data) }
        return hash.finalize().map { String(format:"%02x",$0) }.joined()
    }
    /// Stream every entry without extraction. Credentials, extra/missing entries,
    /// traversal, hard links and altered bytes all prevent lifecycle cleanup.
    public static func verify(archive file: URL, manifest: URL, expectedSHA: String) throws -> Int {
        guard try sha256(file) == expectedSHA else { throw ResearchError("Cloud archive SHA256 mismatch; original sandbox was preserved.") }
        guard let object = try JSONSerialization.jsonObject(with:Data(contentsOf:manifest)) as? [String:Any],object["all_provider_credentials_excluded"] as? Bool == true,let entries = object["entries"] as? [String:[String:Any]] else { throw ResearchError("Invalid recovery manifest.") }
        guard let archive = archive_read_new() else { throw ResearchError("Could not initialize archive verification.") }
        defer { archive_read_free(archive) }
        archive_read_support_filter_all(archive);archive_read_support_format_tar(archive)
        // Enumerate AppleDouble files as real entries rather than silently merging
        // them into their neighbors' metadata on macOS.
        guard archive_read_set_options(archive,"!mac-ext") == ARCHIVE_OK else { throw ResearchError("Could not enable complete archive entry enumeration.") }
        guard archive_read_open_filename(archive,file.path,65_536) == ARCHIVE_OK else { throw ResearchError("Could not read cloud recovery archive.") }
        var seen = Set<String>();var entry: OpaquePointer?
        while true {
            let status = archive_read_next_header(archive,&entry)
            if status == ARCHIVE_EOF { break }
            guard status == ARCHIVE_OK,let entry,let path = archive_entry_pathname(entry) else { throw ResearchError("Damaged recovery archive.") }
            let name = String(cString:path).trimmingCharacters(in:CharacterSet(charactersIn:"/"))
            guard !String(cString:path).hasPrefix("/"),!name.split(separator:"/").contains(".."),!name.hasSuffix("/auth.json"),seen.insert(name).inserted,let expected = entries[name],archive_entry_hardlink(entry) == nil else { throw ResearchError("Unexpected or credential-bearing recovery entry; original sandbox was preserved.") }
            switch expected["kind"] as? String {
            case "file":
                guard archive_entry_filetype(entry) == AE_IFREG,archive_entry_size(entry) == Int64(expected["size"] as? Int ?? -1) else { throw ResearchError("Recovery file metadata mismatch.") }
                var hash = SHA256();var buffer = [UInt8](repeating:0,count:65_536)
                while true { let n = archive_read_data(archive,&buffer,buffer.count);if n == 0 { break };guard n > 0 else { throw ResearchError("Recovery file is truncated.") };hash.update(data:Data(buffer.prefix(n))) }
                guard hash.finalize().map({String(format:"%02x",$0)}).joined() == expected["sha256"] as? String else { throw ResearchError("Recovery file hash mismatch.") }
            case "dir": guard archive_entry_filetype(entry) == AE_IFDIR else { throw ResearchError("Recovery directory mismatch.") }
            case "symlink": guard archive_entry_filetype(entry) == AE_IFLNK,let target = archive_entry_symlink(entry),String(cString:target) == expected["target"] as? String else { throw ResearchError("Recovery symlink mismatch.") }
            default: throw ResearchError("Unsupported recovery entry.")
            }
        }
        guard seen == Set(entries.keys) else { throw ResearchError("Recovery archive entry inventory mismatch (\(seen.count)/\(entries.count)); missing: \(Set(entries.keys).subtracting(seen).sorted().prefix(6).joined(separator: ", ")). The original sandbox was preserved.") }
        return seen.count
    }
}
