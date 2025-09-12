
# Instruções de Uso dos Modelos do NP3

Este repositório fornece um conjunto de ferramentas para instalação, configuração, manipulação de dados e execução de modelos baseados em **PostgreSQL**, **PostGIS** e **Python**.

## Pré-requisitos

Para utilizar os comandos abaixo, certifique-se de ter as seguintes ferramentas instaladas em seu sistema:

- **Ubuntu 20.04** (ou outro sistema baseado em Debian)
- **Python 3.8** ou superior e **pip**
- Permissão para usar comandos com `sudo`

---

## Guia Passo a Passo

### Parte 1: Instalação e Configuração do Banco de Dados
- Busque utilizar as versões dos softwares explicitadas neste tutorial.
1. **Instale o PostgreSQL 12**:
   ```bash
   sudo sh -c 'echo "deb https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
   wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
   sudo apt-get update
   sudo apt-get -y install postgresql-12
   ```

2. **Configure a senha do usuário `postgres`**:
   ```bash
   sudo -u postgres psql
   ```
   - No terminal do PostgreSQL, digite:
     ```sql
     \password postgres
     ```
     Insira a senha `12345` (necessária para os scripts).
   - Saia do terminal PostgreSQL:
     ```sql
     \q
     ```

3. **Instale o PostGIS 3**:
   ```bash
   sudo apt install postgis postgresql-12-postgis-3
   ```

4. **Instale o PgAdmin4**:
   ```bash
   curl -fsS https://www.pgadmin.org/static/packages_pgadmin_org.pub | sudo gpg --dearmor -o /usr/share/keyrings/packages-pgadmin-org.gpg
   sudo sh -c 'echo "deb [signed-by=/usr/share/keyrings/packages-pgadmin-org.gpg] https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" > /etc/apt/sources.list.d/pgadmin4.list && apt update'
   sudo apt install pgadmin4
   ```

5. **Restaure o banco de dados**:
   - Abra o PgAdmin4 e adicione um novo servidor:
     - Nomeie-o como preferir.
     - Na aba **Connection**, insira:
       - **Host name/address**: `localhost`
       - **Password**: `12345`
   - Adicione uma nova database chamada `db_2019`.
   - Clique com o botão direito sobre a database e selecione **Restore**:
     - Escolha o arquivo de backup fornecido.

---

### Parte 2: Extração de Dados e Geração de Arquivos CSV

1. No terminal, navegue até a pasta dos scripts Python de extração de dados.
2. Execute o script principal:
   ```bash
   python3 consolida_dados.py
   ```
3. Para trabalhar com diferentes "potreiros", altere a variável `t.id` no arquivo `selectent.sql`:
   - `1` = p20 infestado
   - `2` = p20 mirapasto
   - `3` = p21 infestado
   - `4` = p21 mirapasto

---

### Parte 3: Execução dos Modelos

1. **Instale as bibliotecas necessárias**:
   ```bash
   pip install numpy matplotlib pandas scikit-learn tensorflow keras-tuner folium branca scipy
   ```
2. Atualize os caminhos e diretórios no script do modelo desejado.
3. No terminal, navegue até a pasta do modelo e execute:
   ```bash
   python3 nome_do_modelo.py
   ```
   - Alguns modelos permitem execução em loop. Informe a quantidade de execuções no terminal.

---

## Trabalhos relacionados ao modelo

Mais informações detalhadas sobre os modelos podem ser encontradas nos trabalhos relacionados:

- **SUPORTE À DECISÃO EM PASTAGENS: ANÁLISE ESPAÇO-TEMPORAL E APRENDIZADO DE MÁQUINA PARA PREDIÇÃO DA DISPONIBILIDADE DE FORRAGEM NO CONTEXTO DE SMART FARMING**: [ACESSE AQUI](https://repositorio.unipampa.edu.br/server/api/core/bitstreams/950bd0ed-b72e-4d06-8a28-0f765ddcdbea/content)  
- **TOUCEIRATECH: UM FARM MANAGEMENT INFORMATION SYSTEM PARA PECUÁRIA DE PRECISÃO BASEADO EM PREDIÇÃO COM REDES NEURAIS RECORRENTES**: [ACESSE AQUI](https://repositorio.unipampa.edu.br/bitstreams/e888f087-4356-4ef2-aef9-d04e07c8affb/download)  
- **IMPACTOS DA FERRAMENTA KERASTUNER EM MODELO DE PREDIÇÃO BASEADO EM LSTM PARA PECUÁRIA SUSTENTÁVEL**: [ACESSE AQUI](https://sol.sbc.org.br/index.php/eradrs/article/view/28012/27822)  
- **ANÁLISE INSTRUMENTAL DE MODELOS RNN PARA AGRICULTURA DIGITAL VISANDO COMPUTAÇÃO VERDE**: [ACESSE AQUI](https://ei.unipampa.edu.br/uploads/evt/arq_trabalhos/29643/etp1_resumo_expandido_29643.pdf)  
- **AJUSTE FINO DE HIPERPARÂMETROS EM MODELO DE PREDIÇÃO PARA AGRICULTURA DIGITAL COM ÁRVORES DE DECISÃO**: [ACESSE AQUI](https://ei.unipampa.edu.br/uploads/evt/arq_trabalhos/29635/etp1_resumo_expandido_29635.pdf)  
- **ÁRVORES DE DECISÃO COMO SOLUÇÃO PARA AJUSTE DE MODELO DE PREDIÇÃO PARA AGRICULTURA DIGITAL**: [ACESSE AQUI](http://www2.bage.ifsul.edu.br/encif/inscricao/pdf/2024110523590338801.pdf)  
- **ANÁLISE DO DESEMPENHO E PROPAGAÇÃO DO ERRO EM MODELOS DE PREDIÇÃO PARA AGRICULTURA DIGITAL SUSTENTÁVEL**: [ACESSE AQUI](https://static.even3.com/anais/982835.pdf?v=638932945030476671)  
- **ANÁLISE DO EFEITO DE ÍNDICES CLIMÁTICOS E PLUVIOMÉTRICOS EM MODELOS DE RNN LSTM PARA PECUÁRIA DE PRECISÃO**: [ACESSE AQUI](https://static.even3.com/anais/996261.pdf?v=638932947430687563)  
- **INSTRUMENTAÇÃO DAS TÉCNICAS ÁRVORE DE DECISÃO E KERASTUNER PARA AUTOAJUSTE DE REDES NEURAIS RECORRENTES GRU NA PERSPECTIVA DE COMPUTAÇÃO SUSTENTÁVEL**: [ACESSE AQUI](https://sol.sbc.org.br/index.php/eradrs/article/view/35364/35153)  


---

## Resultados Esperados

- Após a execução dos modelos, serão gerados:
  - **Arquivos de log** com informações sobre tempo, acurácia e outros detalhes.

---

## Suporte

Para dúvidas ou suporte, entre em contato pelos e-mails:
- **Bianca Durgante**: biancadurgante.aluno@unipampa.edu.br
- **Davi Lemos**: davilemos.aluno@unipampa.edu.br
- **Leonardo Pinho**: leonardopinho@unipampa.edu.br
