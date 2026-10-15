import Foundation
import Security
import ResearchCore

enum DaytonaKeychain {
    static let service = "com.getsupers.autoresearch.daytona"
    static func query(_ account: String) -> [String:Any] { [kSecClass as String:kSecClassGenericPassword,kSecAttrService as String:service,kSecAttrAccount as String:account] }
    static func read(_ account: String) throws -> String {
        var q = query(account);q[kSecReturnData as String] = true;q[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?;let code = SecItemCopyMatching(q as CFDictionary,&result)
        guard code == errSecSuccess,let data = result as? Data,let value = String(data:data,encoding:.utf8) else { throw ResearchError("Enter your Daytona API key in Connections & settings.") }
        return value
    }
    static func save(_ value: String, account: String) throws {
        let q = query(account);let data = Data(value.trimmingCharacters(in:.whitespacesAndNewlines).utf8)
        guard !data.isEmpty else { throw ResearchError("Enter a Daytona API key.") }
        var code = SecItemUpdate(q as CFDictionary,[kSecValueData as String:data] as CFDictionary)
        if code == errSecItemNotFound { var add = q;add[kSecValueData as String] = data;add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly;code = SecItemAdd(add as CFDictionary,nil) }
        guard code == errSecSuccess else { throw ResearchError("macOS Keychain could not save the Daytona key (\(code)).") }
    }
    static func remove(_ account: String) { SecItemDelete(query(account) as CFDictionary) }
}
