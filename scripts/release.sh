#!/bin/bash
# Decimatr Release and Build Script
# Usage: ./scripts/release.sh [command] [options]
#
# Commands:
#   build       Build the package (wheel and sdist)
#   test        Run tests with coverage
#   lint        Run linting and formatting checks
#   format      Format code with ruff
#   clean       Clean build artifacts
#   check       Run all checks (lint, test, build)
#   version     Show or bump version
#   release     Create a release (build, check, tag)
#   publish     Publish to PyPI (or TestPyPI with --test)
#   install     Install package in development mode
#
# Examples:
#   ./scripts/release.sh build
#   ./scripts/release.sh test
#   ./scripts/release.sh version patch    # Bump patch version
#   ./scripts/release.sh release 0.2.0    # Release version 0.2.0
#   ./scripts/release.sh publish --test   # Publish to TestPyPI


#   Usage Examples:

#   # Show all commands
#   ./scripts/release.sh help

#   # Development workflow
#   ./scripts/release.sh install        # Install in dev mode
#   ./scripts/release.sh test -v        # Run tests verbosely
#   ./scripts/release.sh lint           # Check code quality
#   ./scripts/release.sh format         # Auto-format code

#   # Build & Release
#   ./scripts/release.sh build          # Build wheel & sdist
#   ./scripts/release.sh check          # Run all checks (lint, test, build)
#   ./scripts/release.sh clean          # Clean artifacts

#   # Version management
#   ./scripts/release.sh version        # Show current version
#   ./scripts/release.sh version patch  # Bump: 0.1.0 → 0.1.1
#   ./scripts/release.sh version minor  # Bump: 0.1.0 → 0.2.0
#   ./scripts/release.sh version major  # Bump: 0.1.0 → 1.0.0
#   ./scripts/release.sh version 0.2.0  # Set to specific version

#   # Release workflow
#   ./scripts/release.sh release 0.2.0  # Create release (updates version, runs checks, creates git tag)
#   ./scripts/release.sh publish --test # Publish to TestPyPI
#   ./scripts/release.sh publish        # Publish to PyPI

#   Features:
#   - Colored terminal output
#   - Version bumping (major/minor/patch)
#   - Automatic version sync between pyproject.toml and __init__.py
#   - Pre-release checks (lint, test, build)
#   - Git tag creation
#   - TestPyPI and PyPI publishing
#   - Clean build artifacts

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Helper functions
print_header() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Get current version from pyproject.toml
get_version() {
    grep -Po '(?<=^version = ")[^"]*' pyproject.toml
}

# Update version in both pyproject.toml and __init__.py
set_version() {
    local new_version=$1

    # Update pyproject.toml
    sed -i "s/^version = \".*\"/version = \"$new_version\"/" pyproject.toml

    # Update __init__.py
    sed -i "s/__version__ = \".*\"/__version__ = \"$new_version\"/" decimatr/__init__.py

    print_success "Version updated to $new_version"
}

# Bump version (major, minor, patch)
bump_version() {
    local bump_type=$1
    local current_version=$(get_version)

    IFS='.' read -r major minor patch <<< "$current_version"

    case $bump_type in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
        *)
            print_error "Invalid bump type: $bump_type (use major, minor, or patch)"
            exit 1
            ;;
    esac

    local new_version="$major.$minor.$patch"
    set_version "$new_version"
    echo "$new_version"
}

# Commands
cmd_build() {
    print_header "Building Package"

    # Clean previous builds
    rm -rf dist/ build/ *.egg-info

    # Build
    python -m build

    # Check
    twine check dist/*

    print_success "Build complete!"
    echo ""
    echo "Artifacts:"
    ls -lh dist/
}

cmd_test() {
    print_header "Running Tests"

    local verbose=""
    if [[ "$1" == "-v" || "$1" == "--verbose" ]]; then
        verbose="-v"
    fi

    pytest tests/ $verbose --cov=decimatr --cov-report=term-missing --cov-report=xml

    print_success "Tests complete!"
}

cmd_lint() {
    print_header "Running Linting Checks"

    local exit_code=0

    echo "Checking with ruff..."
    if ruff check decimatr tests; then
        print_success "Ruff check passed"
    else
        print_warning "Ruff found issues (non-blocking)"
    fi

    echo ""
    echo "Checking formatting with ruff..."
    if ruff format --check decimatr tests; then
        print_success "Ruff format check passed"
    else
        print_error "Ruff format check failed"
        exit_code=1
    fi

    echo ""
    echo "Running mypy type check..."
    if mypy decimatr --ignore-missing-imports; then
        print_success "Mypy check passed"
    else
        print_warning "Mypy found issues (non-blocking)"
    fi

    if [[ $exit_code -eq 0 ]]; then
        print_success "All lint checks passed!"
    else
        print_error "Some lint checks failed"
        exit $exit_code
    fi
}

cmd_format() {
    print_header "Formatting Code"

    echo "Running ruff with auto-fix..."
    ruff check decimatr tests --fix || true

    echo ""
    echo "Running ruff format..."
    ruff format decimatr tests

    print_success "Formatting complete!"
}

cmd_clean() {
    print_header "Cleaning Build Artifacts"

    rm -rf dist/
    rm -rf build/
    rm -rf *.egg-info
    rm -rf .pytest_cache/
    rm -rf .mypy_cache/
    rm -rf .ruff_cache/
    rm -rf .coverage
    rm -rf coverage.xml
    rm -rf htmlcov/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

    print_success "Clean complete!"
}

cmd_check() {
    print_header "Running All Checks"

    cmd_lint
    echo ""
    cmd_test
    echo ""
    cmd_build

    print_success "All checks passed!"
}

cmd_version() {
    local action=$1

    if [[ -z "$action" ]]; then
        echo "Current version: $(get_version)"
    elif [[ "$action" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        # Direct version set
        set_version "$action"
    elif [[ "$action" == "major" || "$action" == "minor" || "$action" == "patch" ]]; then
        # Bump version
        local new_version=$(bump_version "$action")
        echo "New version: $new_version"
    else
        print_error "Invalid version action: $action"
        echo "Usage: $0 version [major|minor|patch|X.Y.Z]"
        exit 1
    fi
}

cmd_release() {
    local version=$1

    if [[ -z "$version" ]]; then
        print_error "Version required for release"
        echo "Usage: $0 release <version>"
        echo "Example: $0 release 0.2.0"
        exit 1
    fi

    print_header "Creating Release v$version"

    # Check for uncommitted changes
    if [[ -n $(git status --porcelain) ]]; then
        print_error "There are uncommitted changes. Please commit or stash them first."
        exit 1
    fi

    # Update version
    set_version "$version"

    # Run all checks
    cmd_check

    # Commit version change
    git add pyproject.toml decimatr/__init__.py
    git commit -m "chore: bump version to $version"

    # Create tag
    git tag -a "v$version" -m "Release v$version"

    print_success "Release v$version created!"
    echo ""
    echo "Next steps:"
    echo "  1. Review the changes: git log --oneline -5"
    echo "  2. Push to remote: git push origin master --tags"
    echo "  3. Create GitHub release at: https://github.com/DylanLIiii/decimatr/releases/new"
    echo "  4. Or publish manually: $0 publish"
}

cmd_publish() {
    local test_pypi=false

    if [[ "$1" == "--test" || "$1" == "-t" ]]; then
        test_pypi=true
    fi

    # Check if dist exists
    if [[ ! -d "dist" || -z "$(ls -A dist)" ]]; then
        print_warning "No build artifacts found. Building first..."
        cmd_build
    fi

    if $test_pypi; then
        print_header "Publishing to TestPyPI"
        twine upload --repository testpypi dist/*

        print_success "Published to TestPyPI!"
        echo ""
        echo "Test installation:"
        echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ decimatr"
    else
        print_header "Publishing to PyPI"

        echo -e "${YELLOW}Warning: This will publish to the real PyPI!${NC}"
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi

        twine upload dist/*

        print_success "Published to PyPI!"
        echo ""
        echo "Install with:"
        echo "  pip install decimatr"
    fi
}

cmd_install() {
    print_header "Installing in Development Mode"

    local extras=""
    if [[ "$1" == "--gpu" ]]; then
        extras="[dev,gpu]"
    else
        extras="[dev]"
    fi

    pip install -e "$extras"

    print_success "Installation complete!"
}

cmd_help() {
    echo "Decimatr Release and Build Script"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  build              Build the package (wheel and sdist)"
    echo "  test [-v]          Run tests with coverage"
    echo "  lint               Run linting and formatting checks"
    echo "  format             Format code with ruff"
    echo "  clean              Clean build artifacts"
    echo "  check              Run all checks (lint, test, build)"
    echo "  version [action]   Show or bump version (major|minor|patch|X.Y.Z)"
    echo "  release <version>  Create a release (build, check, tag)"
    echo "  publish [--test]   Publish to PyPI (or TestPyPI with --test)"
    echo "  install [--gpu]    Install package in development mode"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build                  # Build the package"
    echo "  $0 test -v                # Run tests verbosely"
    echo "  $0 version                # Show current version"
    echo "  $0 version patch          # Bump patch version (0.1.0 -> 0.1.1)"
    echo "  $0 version 0.2.0          # Set version to 0.2.0"
    echo "  $0 release 0.2.0          # Create release 0.2.0"
    echo "  $0 publish --test         # Publish to TestPyPI"
    echo "  $0 publish                # Publish to PyPI"
}

# Main
main() {
    local command=$1
    shift || true

    case $command in
        build)
            cmd_build "$@"
            ;;
        test)
            cmd_test "$@"
            ;;
        lint)
            cmd_lint "$@"
            ;;
        format)
            cmd_format "$@"
            ;;
        clean)
            cmd_clean "$@"
            ;;
        check)
            cmd_check "$@"
            ;;
        version)
            cmd_version "$@"
            ;;
        release)
            cmd_release "$@"
            ;;
        publish)
            cmd_publish "$@"
            ;;
        install)
            cmd_install "$@"
            ;;
        help|--help|-h|"")
            cmd_help
            ;;
        *)
            print_error "Unknown command: $command"
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
