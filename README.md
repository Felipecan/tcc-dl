## Pasta destinada aos códigos relacionados ao TCC

### Configurando e instalando as dependências necessárias para executar os códigos:

Instale o [Minconda3](https://conda.io/en/latest/miniconda.html) de acordo com o seu sistema.

Clone o repositório atual, dentro da pasta [codes]() ira conter o arquivo [dl-env.yml](dl-env.yml), que descreve as configurações de ambiente que será usada.

No terminal, dê o seguinte comando para criar o ambiente:

```sh
$ conda env create --file dl-env.yml
```

Após criado o ambiente, para ativá-lo use o comando no terminal do Linux:
```sh
$ conda activate tcc-env-cpu
```

Caso tenha configurado a placa de vídeo e deseje utilizar a GPU, execute o primeiro comando (para criar o ambiente), mas utilizando o arquivo [dl-env-gpu.yml](dl-env-gpu.yml). Apos isso, para ativar o ambiente: 
```sh
$ conda activate tcc-env-gpu
```
### As funcionalidades dos scripts e como executá-los

[pre_processing.py](pre_processing.py): Esse arquivo faz o pre processamento dos dados que serão usados na rede. Ele separa os dados do csv para cada áudio e obtem os espectrogramas dos mesmo. 

Para executar o script:
```sh
$ python pre_processing.py --csv caminho/para/arquivo.csv --audio_folders caminho/para/pastas_audios
```

[VGG19.py](VGG19.py): Esse arquivo é uma classe que contém a implementação da rede convolucional VGG19 apresentada por *Simonyan and Zisserman* em 2014.

[main.py](main.py): Esse script tem como função executar todas as tarefas até agora apresentada. Ele funciona de duas maneiras: chamando as funções da [VGG19](VGG19.py) ou combinado com o [pré-processamento](pre_processing.py) para pré-processar os dados e treinar a rede de uma vez só.

Para executar o script [main.py](main.py) somente como treino (os dados devem está pre-processados):
```sh
$ python main.py --spect_folders /caminho/para/pastas_espectrogramas --mode training
```

Para executar o script [main.py](main.py) combinando o pré-processamento com o treinamento:
```sh
$ python main.py --mode all_proc --csv caminho/para/arquivo.csv --audio_folders caminho/para/pastas_audios 
```

Obs 1: O nome do ambiente pode ser mudado no arquivo [dl-env.yml](dl-env.yml), primeira linha.
