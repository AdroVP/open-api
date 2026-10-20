### Crear un env
```
python -m venv env
```

### Activar el env

#### Windows
```
env\Scripts\activate
```

#### Linux / Mac
```
source env/bin/activate
```

### Construir la imagen de docker
```
docker build -t pose-api .
```

### Ejecutar el contenedor
```
docker run -d -p 8000:8000 --name pose-api-service pose-api
```

### Verificar estado:
[http://localhost:8000/docs](http://localhost:8000/docs)

### Ejecución de tests
```
docker run --rm pose-api pytest
```

### Enviar peticion a /post
```
curl -X POST "http://localhost:8000/pose" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@imagen.jpg"
```