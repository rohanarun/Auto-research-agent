#!/bin/zsh
set -euo pipefail
PROJECT_ROOT="${0:A:h:h}"
cd "$PROJECT_ROOT"
python3 scripts/check_workspace.py "$PROJECT_ROOT"
: "${RESEARCH_SIGNING_IDENTITY:?Set a Developer ID Application signing identity}"
: "${RESEARCH_NOTARY_PROFILE:?Set an existing notarytool keychain profile}"
export TMPDIR="$PROJECT_ROOT/tmp/"
RELEASE="$PROJECT_ROOT/build/distribution"
export RESEARCH_APP_OUTPUT="$RELEASE/staging/Auto Research.app"
./scripts/build.sh
# Public artifacts contain only blank installation defaults; no local paths or runtime data.
cp Resources/Defaults.example.json "$RESEARCH_APP_OUTPUT/Contents/Resources/Defaults.json"
cp LICENSE "$RESEARCH_APP_OUTPUT/Contents/Resources/LICENSE"
chmod 644 "$RESEARCH_APP_OUTPUT/Contents/Resources/Defaults.json"
codesign --force --options runtime --timestamp --sign "$RESEARCH_SIGNING_IDENTITY" "$RESEARCH_APP_OUTPUT"
codesign --verify --deep --strict "$RESEARCH_APP_OUTPUT"
ZIP="$RELEASE/Auto-research-agent-notarization.zip"
ditto -c -k --keepParent "$RESEARCH_APP_OUTPUT" "$ZIP"
xcrun notarytool submit "$ZIP" --keychain-profile "$RESEARCH_NOTARY_PROFILE" --wait --output-format json > "$RELEASE/app-notarization.json"
python3 - "$RELEASE/app-notarization.json" <<'PY'
import json,sys
receipt=json.load(open(sys.argv[1]))
assert receipt.get('status')=='Accepted', receipt
PY
xcrun stapler staple "$RESEARCH_APP_OUTPUT"
xcrun stapler validate "$RESEARCH_APP_OUTPUT"
spctl --assess --type execute --verbose=2 "$RESEARCH_APP_OUTPUT"
ln -sfn /Applications "$RELEASE/staging/Applications"
cat > "$RELEASE/staging/Read me.txt" <<'TXT'
Auto-research agent

Requires macOS 14 or newer and Apple Silicon.
Drag Auto Research into Applications, then open it.
Choose a folder on a mounted external drive for research projects and chat history.
Install Codex or Claude Code and sign in through Connections & settings.
If a CLI is not detected, enter its executable path in Connections & settings.

Source and MIT license: https://github.com/rohanarun/Auto-research-agent
TXT
VERSION=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$RESEARCH_APP_OUTPUT/Contents/Info.plist")
DMG="$RELEASE/Auto-research-agent-$VERSION-arm64.dmg"
hdiutil create -volname 'Auto-research agent' -srcfolder "$RELEASE/staging" -format UDZO -ov "$DMG"
codesign --force --timestamp --sign "$RESEARCH_SIGNING_IDENTITY" "$DMG"
xcrun notarytool submit "$DMG" --keychain-profile "$RESEARCH_NOTARY_PROFILE" --wait --output-format json > "$RELEASE/dmg-notarization.json"
python3 - "$RELEASE/dmg-notarization.json" <<'PY'
import json,sys
receipt=json.load(open(sys.argv[1]))
assert receipt.get('status')=='Accepted', receipt
PY
xcrun stapler staple "$DMG"
xcrun stapler validate "$DMG"
hdiutil verify "$DMG"
(cd "$RELEASE" && shasum -a 256 "${DMG:t}") > "$DMG.sha256"
echo "$DMG"
