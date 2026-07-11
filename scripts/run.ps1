param(
    [string]$Command = "dev"
)

switch ($Command) {
    "dev" {
        Write-Host "Starting development server..." -ForegroundColor Green
        uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    }
    "docker" {
        Write-Host "Starting Docker Compose..." -ForegroundColor Green
        docker-compose up --build
    }
    "migrate" {
        Write-Host "Running Alembic migrations..." -ForegroundColor Green
        alembic upgrade head
    }
    "makemigrations" {
        param([string]$Message = "auto")
        Write-Host "Creating new migration..." -ForegroundColor Green
        alembic revision --autogenerate -m $Message
    }
    "test" {
        Write-Host "Running tests..." -ForegroundColor Green
        pytest tests/ -v
    }
    "lint" {
        Write-Host "Running ruff linter..." -ForegroundColor Green
        ruff check .
    }
    default {
        Write-Host "Unknown command. Use: dev, docker, migrate, makemigrations, test, lint" -ForegroundColor Yellow
    }
}
