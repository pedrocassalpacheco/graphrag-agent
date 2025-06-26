# GraphRAG Agent

This project implements an agent using Google's Agent Development Kit (ADK).

## Project Structure

```
.
├── src/                    # Source code directory
│   ├── agent/             # Core agent implementation
│   ├── config/            # Configuration files and settings
│   ├── utils/             # Utility functions and helpers
│   ├── models/            # Data models and schemas
│   ├── handlers/          # Request/response handlers
│   └── services/          # External service integrations
├── tests/                 # Test directory
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── docs/                  # Documentation
└── examples/              # Example implementations
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development

- Place your agent implementation in `src/agent/`
- Add configuration in `src/config/`
- Implement handlers in `src/handlers/`
- Add utility functions in `src/utils/`
- Define data models in `src/models/`
- Implement service integrations in `src/services/`

## Testing

Run unit tests:
```bash
python -m pytest tests/unit
```

Run integration tests:
```bash
python -m pytest tests/integration
```

## Documentation

Detailed documentation can be found in the `docs/` directory.

## License

[Add your license information here]
