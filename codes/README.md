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
$ conda activate tcc-dl-env
```

[pre_processing.py](pre_processing.py): Esse arquivo faz o pre processamento dos dados que serão usados na rede. Ele separa os dados do csv para cada áudio e obtem os espectrogramas dos mesmo. 

Para executar os scripts:
```sh
$ python pre_processing.py --csv caminho/para/arquivo.csv --audio_folders caminho/para/pastas_audios
```

Obs 1: O nome do ambiente pode ser mudado no arquivo [dl-env.yml](dl-env.yml), primeira linha.
