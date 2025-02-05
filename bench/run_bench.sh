# Set the repository URL and target directory
REPO_URL="https://github.com/coseal/aslib_data.git"
TARGET_DIR="bench/aslib_data"

# Check if the directory exists and contains a git repository
if [ -d "$TARGET_DIR/.git" ]; then
    echo "Repository already exists at $TARGET_DIR."
else
    echo "Cloning repository..."
    git clone "$REPO_URL" "$TARGET_DIR"
    echo "Repository cloned to $TARGET_DIR."
fi



