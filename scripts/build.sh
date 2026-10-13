#!/bin/zsh
set -euo pipefail
PROJECT_ROOT="${0:A:h:h}"
export TMPDIR="$PROJECT_ROOT/tmp/"
export CLANG_MODULE_CACHE_PATH="$PROJECT_ROOT/build-cache/clang"
export SWIFT_MODULECACHE_PATH="$PROJECT_ROOT/build-cache/swift"
cd "$PROJECT_ROOT"
python3 scripts/check_workspace.py "$PROJECT_ROOT"
PYTHONDONTWRITEBYTECODE=1 python3 scripts/configure.py
mkdir -p "$TMPDIR" "$CLANG_MODULE_CACHE_PATH" "$SWIFT_MODULECACHE_PATH"
swift build -c release --scratch-path "$PROJECT_ROOT/.build" --cache-path "$PROJECT_ROOT/build-cache/spm" --disable-sandbox
APP="${RESEARCH_APP_OUTPUT:-$PROJECT_ROOT/build/Auto Research.app}"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$PROJECT_ROOT/.build/release/AutoResearch" "$APP/Contents/MacOS/AutoResearch.new"
mv "$APP/Contents/MacOS/AutoResearch.new" "$APP/Contents/MacOS/AutoResearch"
cp "$PROJECT_ROOT/Resources/Defaults.json" "$PROJECT_ROOT/Resources/ResearchPolicy.md" "$PROJECT_ROOT/Resources/ContinuePolicy.md" "$PROJECT_ROOT/Resources/catalog-source.html" "$APP/Contents/Resources/"
cp "$PROJECT_ROOT/Resources/CatalogPolicy.json" "$APP/Contents/Resources/"
cp "$PROJECT_ROOT/Resources/SharedContextDefaults.json" "$PROJECT_ROOT/Resources/SharedAgentPolicy.md" "$PROJECT_ROOT/Resources/SharedContextPolicy.md" "$APP/Contents/Resources/"
cp "$PROJECT_ROOT/Resources/Info.plist" "$APP/Contents/Info.plist"
if [[ -f "$PROJECT_ROOT/Resources/AppIcon.icns" ]]; then cp "$PROJECT_ROOT/Resources/AppIcon.icns" "$APP/Contents/Resources/"; fi
codesign --force --sign - "$APP"
echo "$APP"
