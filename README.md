# HOWTO  
## run ollama in docker and pull model  

```
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama  
docker exec -it ollama ollama pull gemma3:270m
```
use env variable `OLLAMA_BASE_URL=http://localhost:11434`  


refs:  
https://hub.docker.com/r/ollama/ollama   