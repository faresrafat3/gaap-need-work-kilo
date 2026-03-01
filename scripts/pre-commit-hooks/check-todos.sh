#!/usr/bin/env bash
# Check for TODO/FIXME/XXX comments in staged files

set -euo pipefail

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

found_issues=0

for file in "$@"; do
    # Skip non-Python files
    if [[ ! "$file" =~ \.py$ ]]; then
        continue
    fi

    # Check for TODO/FIXME/XXX (but not in comments that are already marked)
    if grep -n -H -E '^\s*#[[:space:]]*(TODO|FIXME|XXX|HACK|BUG)' "$file" 2>/dev/null || \
       grep -n -H -E '""".*(TODO|FIXME|XXX|HACK|BUG).*"""' "$file" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Found TODO/FIXME markers in $file${NC}"
        found_issues=$((found_issues + 1))
    fi
done

if [ $found_issues -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Consider addressing these markers before committing.${NC}"
    echo "This is a warning and will not block the commit."
    # Return 0 to not block the commit, just warn
    exit 0
fi

exit 0
