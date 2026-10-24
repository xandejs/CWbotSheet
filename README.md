# CWbotSheet

Bot de análise e geração de sinais para o Craft World, com foco em identificar oportunidades de compra e venda a partir de preços, indicadores técnicos e da recomendação da API do marketplace.

O projeto lê a aba `Dados` de uma planilha do Google Sheets, normaliza o histórico em grade temporal de 15 minutos, calcula indicadores, treina um modelo XGBoost por item e grava os resultados em abas como `Sinais_IA`, `Sinais_Hoje`, `XGB_Audit`, `Cerebro_RL`, `Cerebro_RL_Item` e `Alocacao_Recursos`.

## O que o bot faz

- Coleta e padroniza os preços por item em uma grade uniforme
- Calcula indicadores como média macro, desvio, RSI, lags e variação de 72h
- Combina os sinais do mercado com a recomendação da API do jogo
- Gera sinais de compra e venda para monitoramento
- Usa um modelo XGBoost para prever a tendência de preço
- Aplica lógica de Q-learning para filtrar sinais e ajustar decisões
- Produz um ranking de itens para alocação de recursos
- Envia alertas por Telegram quando um sinal relevante é gerado

## Fluxo principal

1. Lê a aba `Dados` da planilha
2. Normaliza o histórico para intervalos de 15 minutos
3. Calcula os indicadores por item
4. Carrega ou treina o modelo pooled do XGBoost
5. Avalia cada item em relação ao preço atual, alvo e risco
6. Gera sinais em `Sinais_IA`
7. Atualiza o dashboard de estado em `Sinais_Hoje`
8. Persiste o histórico de auditoria em `XGB_Audit`
9. Atualiza os estados do cérebro RL e as recomendações de alocação

## Abas geradas pelo bot

- `Dados`: fonte principal, leitura apenas
- `Sinais_IA`: histórico de sinais com status, preço de entrada, referência financeira, lucro desejado e concordância com a API
- `Sinais_Hoje`: snapshot do estado atual de todos os itens
- `XGB_Audit`: métricas de desempenho do modelo
- `Cerebro_RL`: Q-table global
- `Cerebro_RL_Item`: Q-table por item
- `Alocacao_Recursos`: ranking de itens com melhor potencial para exposição

## Tecnologias usadas

- Python
- pandas
- numpy
- xgboost
- gspread
- oauth2client
- gspread-formatting
- requests

## Requisitos

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Credenciais

O bot usa uma conta de serviço do Google para acessar a planilha.

### Opção 1: variável de ambiente

```bash
export GOOGLE_CREDENTIALS_JSON='{"type": "service_account", ...}'
```

### Opção 2: arquivo local

Crie um arquivo `credenciais_robo.json` com as credenciais da service account.

## Telegram

Para envio de alertas, defina estas variáveis de ambiente:

```bash
export TELEGRAM_BOT_TOKEN="seu_token"
export TELEGRAM_CHAT_ID="seu_chat_id"
```

Se não estiverem configuradas, o bot apenas imprime mensagens no console.

## Estrutura do repositório

```text
.
├── bot_craft_world.py   # script principal do bot
├── requirements.txt      # dependências Python
├── README.md            # documentação do projeto
├── cache_craft_world    # cache de indicadores e modelos
└── credenciais_robo.json # credenciais locais (se usar esta opção)
```

## Como executar

```bash
python bot_craft_world.py
```

O script acessa a planilha configurada em `SPREADSHEET_ID`, atualiza as abas internas e salva o estado em cache para evitar recalcular tudo a cada ciclo.

## Observações importantes

- O projeto foi adaptado para o Craft World, onde o foco é análise de preço e geração de sinais, sem depender de estoque real.
- A grade de tempo é de 15 minutos e todas as janelas são calculadas em tempo, não em linhas fixas de dados.
- O modelo e os parâmetros podem ser ajustados conforme o histórico real da planilha.
- A lógica de alocação e o scoring de risco foram pensados para priorizar itens com melhor histórico de acerto e retorno por dia.

## Objetivo

O bot serve como ferramenta de apoio para acompanhamento do mercado do Craft World, com geração de sinais, validação histórica e ranking de oportunidades para exposição.

## Licença

Este projeto está sem licença explícita no repositório. Verifique com o mantenedor antes de usar em produção ou redistribuir.
