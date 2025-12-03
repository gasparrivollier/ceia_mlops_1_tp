# Usamos una imagen ligera de Node
FROM node:24.11.1-alpine
# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos los archivos de dependencia
# Nota: Ajustamos el path porque el contexto será la carpeta 'frontend'
COPY package*.json ./

# Instalamos dependencias
RUN npm install

# Copiamos el resto del código fuente
COPY . .

# Exponemos el puerto de Vite (generalmente 5173)
EXPOSE 5173

# Comando para levantar en modo desarrollo (permite cambios en caliente)
CMD ["npm", "run", "dev", "--", "--host"]