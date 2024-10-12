from invoke import task
import os
import shutil
import subprocess


@task
def install(c):
    """Instala las dependencias del proyecto."""
    c.run("pip install -r requirements.txt")
    print("Dependencias instaladas correctamente.")


@task
def uninstall(c):
    """Desinstala las dependencias del proyecto."""
    with open("requirements.txt", "r") as f:
        packages = f.read().splitlines()
    for package in packages:
        c.run(f"pip uninstall -y {package}")
    print("Dependencias desinstaladas correctamente.")


@task
def update(c):
    """Actualiza las dependencias del proyecto."""
    c.run("pip install --upgrade -r requirements.txt")
    print("Dependencias actualizadas correctamente.")


@task
def start(c):
    """Inicia el servidor en modo producción."""
    print("Iniciando servidor en modo producción...")
    print("Servidor corriendo en http://localhost:80")
    subprocess.run(["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"])


@task
def dev(c):
    """Inicia el servidor en modo desarrollo con recarga automática."""
    print("Iniciando servidor en modo desarrollo...")
    print("Servidor corriendo en http://localhost:8000")
    subprocess.run(
        ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    )


@task
def clean(c):
    """Limpia archivos temporales y caches."""
    patterns = ["__pycache__", "*.pyc"]
    for root, dirs, files in os.walk("."):
        for pattern in patterns:
            if pattern in dirs:
                shutil.rmtree(os.path.join(root, pattern))
            for file in files:
                if file.endswith(".pyc"):
                    os.remove(os.path.join(root, file))
    print("Limpieza completada.")


@task
def list(c):
    """Lista las dependencias instaladas."""
    c.run("pip list")


@task
def predict(c):
    """Predicción de prueba."""
    c.run("python testing/predict.py")
