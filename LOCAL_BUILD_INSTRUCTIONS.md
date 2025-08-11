# Local Build Instructions for Scaling Book

## Prerequisites

### On Windows (using WSL2 or Git Bash)
1. Install Ruby (recommended version 3.0+)
2. Install Bundler
3. Install Git

### On macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Ruby and Bundler
brew install ruby
gem install bundler
```

### On Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt-get update

# Install Ruby and dependencies
sudo apt-get install -y ruby-full ruby-bundler build-essential zlib1g-dev

# For Jekyll specifically, you might also need:
sudo apt-get install -y nodejs npm
```

## Building the Project

### Step 1: Navigate to the project directory
```bash
cd /path/to/scaling-book
```

### Step 2: Install dependencies
```bash
bundle install
```

If you encounter permission issues, try:
```bash
bundle install --path vendor/bundle
```

### Step 3: Serve the site locally
```bash
bundle exec jekyll serve
```

Or with more options:
```bash
# Serve with live reload and open browser
bundle exec jekyll serve --livereload --open-url

# Serve on a specific port
bundle exec jekyll serve --port 4001

# Serve with baseurl (important for GitHub Pages)
bundle exec jekyll serve --baseurl "/scaling-book"
```

### Step 4: Access the site
Open your browser and navigate to:
- http://localhost:4000/scaling-book/ (default)
- Or the port you specified

## Common Issues and Solutions

### Issue: "Could not find gem"
Solution:
```bash
bundle update
bundle install
```

### Issue: "Permission denied"
Solution:
```bash
# Install gems locally
bundle config set --local path 'vendor/bundle'
bundle install
```

### Issue: "JavaScript heap out of memory"
Solution:
```bash
# Increase Node memory limit
export NODE_OPTIONS="--max_old_space_size=4096"
bundle exec jekyll serve
```

### Issue: "Address already in use"
Solution:
```bash
# Use a different port
bundle exec jekyll serve --port 4001
```

### Issue: ImageMagick not found (for responsive images)
Solution:
```bash
# On macOS
brew install imagemagick

# On Linux
sudo apt-get install imagemagick

# On Windows (WSL)
sudo apt-get install imagemagick
```

## Docker Alternative

If you prefer using Docker, the project includes Docker support:

### Using Docker Compose
```bash
# Build and serve
docker-compose up

# Or use the slim version
docker-compose -f docker-compose-slim.yml up
```

### Access the site
Navigate to: http://localhost:8080/scaling-book/

## Development Tips

### Watch for changes
The site automatically rebuilds when you modify files (except _config.yml)

### Clear cache if needed
```bash
bundle exec jekyll clean
bundle exec jekyll serve
```

### Build for production
```bash
JEKYLL_ENV=production bundle exec jekyll build
```

### Check the Chinese version
After starting the server, click the "中文" button in the navigation bar to switch to the Chinese version.

## File Structure for Language Switch

The language switch feature uses:
- `_includes/language-switcher.liquid` - Language toggle component
- `_includes/header.liquid` - Modified to include the language switcher
- `Chinese version/` directory - Contains all Chinese translations

## Testing the Language Switch

1. Start the local server
2. Navigate to any page
3. Click the "中文" button in the top navigation
4. The page should switch to the Chinese version
5. Click "English" to switch back

## Troubleshooting Language Switch

If the language switch doesn't work:
1. Check browser console for JavaScript errors
2. Ensure localStorage is enabled in your browser
3. Clear browser cache and reload
4. Verify the Chinese version files exist in the `Chinese version/` directory

## Support

For more help:
- Check the project's README.md
- Review the Jekyll documentation: https://jekyllrb.com/docs/
- Check the al-folio theme documentation: https://github.com/alshedivat/al-folio