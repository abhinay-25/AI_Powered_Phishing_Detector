# VS Code + Copilot + Colab Workflow

1. Develop with Copilot in VS Code
   - Write and edit code under `src/` and notebooks under `notebooks/`.
   - Use Copilot inline completions for faster iteration.

2. Commit and push changes
```
git add .
git commit -m "Update feature extraction logic"
git push origin main
```

3. In Google Colab
```
!git pull origin main
# Run notebooks and train models
```

4. Optional: Branching
```
git checkout -b feature/improve-features
# ... work ...
git push -u origin feature/improve-features
```

5. Sync back to VS Code
- Pull latest changes in VS Code Source Control.

Tips
- Keep data files out of Git; store in cloud or use small samples.
- Use `requirements.txt` to keep Colab and local envs consistent.
