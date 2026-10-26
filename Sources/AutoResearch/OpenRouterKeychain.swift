import Foundation
import Security
import LocalAuthentication
import ResearchCore

protocol OpenRouterCredentialStore {
    func containsKey() -> Bool
    func read() throws -> String
    func save(_ value: String) throws
    func remove() throws
}

struct OpenRouterKeychain: OpenRouterCredentialStore {
    private var query: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "com.getsupers.autoresearch.openrouter", kSecAttrAccount as String: "api-key"]
    }
    func containsKey() -> Bool {
        var q = query
        q[kSecReturnAttributes as String] = true
        let context = LAContext(); context.interactionNotAllowed = true
        q[kSecUseAuthenticationContext as String] = context
        return SecItemCopyMatching(q as CFDictionary, nil) == errSecSuccess
    }
    func read() throws -> String {
        var q = query; q[kSecReturnData as String] = true; q[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data, let value = String(data: data, encoding: .utf8), !value.isEmpty else {
            throw ResearchError("Save an OpenRouter API key in Connections → Gateway before starting this run.")
        }
        return value
    }
    func save(_ value: String) throws {
        let data = Data(value.utf8)
        var code = SecItemUpdate(query as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if code == errSecItemNotFound {
            var q = query; q[kSecValueData as String] = data; q[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
            code = SecItemAdd(q as CFDictionary, nil)
        }
        guard code == errSecSuccess else { throw ResearchError("macOS Keychain could not save the OpenRouter key (\(code)).") }
    }
    func remove() throws {
        let code = SecItemDelete(query as CFDictionary)
        guard code == errSecSuccess || code == errSecItemNotFound else { throw ResearchError("macOS Keychain could not remove the OpenRouter key (\(code)).") }
    }
}
