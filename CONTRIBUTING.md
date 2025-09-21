# Contributing to Churn-X
Thanks for your interest in contributing! We welcome improvements, bug fixes, tests, and documentation updates.

## How to contribute
### 1. Fork the repository
Create your own copy of this repo by clicking ```Fork```

### 2. Clone your fork
```
git clone https://github.com/<your-username>/churn-x.git
```

```
cd churn-x
```

### 3. Create a branch
Use a descriptive branch name:
```
git checkout -b fix/data-ingestion-bug
```

### 4. Install dependencies
```
uv sync
```

### 5. Run tests locally
```
pytest tests/ --maxfail=1 --disable-warnings -q
```

### 6. Commit your changes
```
git commit -m "Fix: handled empty dataframe in data ingestion"
```

### 7. Push your branch
```
git push origin fix/data-ingestion-bug
```

### 8. Open a Pull Request (PR)
Go to your fork on GitHub and open a PR against the ```main``` branch.

## Contributing Guidelines
- Code style: Follow PEP8 [PEP8](https://peps.python.org/pep-0008/)
- Testing: Write/Update tests for any code changes (```pytest```)
- Docs: Update docstrings & README if you add features
- Commits: Use clear commit messages (```Fix:```, ```Feat:```, ```Docs:```, ```Test:```)
- PR reviews: Be open to feedback and iterate on your PR

## Community
- Found a bug? Open an issue
- Want a feature? Create a discussion or PR.
- New to open source? Start with ```good first issue```.