from invoke import task


@task
def install(c):
    """Instala las dependencias del proyecto."""
    c.run("pip install -r requirements.txt")


@task
def update(c):
    """Actualiza las dependencias del proyecto."""
    c.run("pip install --upgrade -r requirements.txt")


@task
def start(c):
    """Inicia el servidor en modo producción."""
    c.run("uvicorn app.main:app --host 0.0.0.0 --port 80")


@task
def dev(c):
    """Inicia el servidor en modo desarrollo con recarga automática."""
    c.run("uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")


@task
def clean(c):
    """Limpia archivos temporales y caches."""
    c.run("find . -type d -name __pycache__ -exec rm -rf {} +")
    c.run("find . -type f -name '*.pyc' -delete")
