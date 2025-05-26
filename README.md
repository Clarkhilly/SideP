# UPDATE YOUR GITHUB REPOSITORY (NOT THE LIVE WEBSITE)

# Step 1: Navigate to your project root
cd C:\Users\clark\Desktop\SideP

# Step 2: Add your changes
git add .
# Or add specific files:
# git add TS/src/index.tsx
# git add Python/new_script.py
# git add UnderVal/Undervalue.py

# Step 3: Commit your changes
git commit -m "Description of what you changed"

# Step 4: Push to GitHub
git push origin main

# EXAMPLES:

# If you update your Python stock analysis:
git add UnderVal/Undervalue.py
git commit -m "Improved stock analysis algorithm"
git push origin main

# If you add a new Python project:
git add Python/new_project/
git commit -m "Added new machine learning project"
git push origin main

# If you modify your React website code:
git add TS/src/
git commit -m "Updated website styling"
git push origin main

# TO UPDATE THE LIVE WEBSITE (only if you changed React code):
cd TS
npm run build
npm run deploy

# QUICK SUMMARY:
# git add → git commit → git push = Updates GitHub repository
# npm run deploy = Updates live website at https://Clarkhilly.github.io/SideP
