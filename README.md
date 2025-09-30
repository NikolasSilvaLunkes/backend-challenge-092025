
# Faça as etapas do teste e utilize o seguinte comando para testar**
```
RUN_PERF=1 pytest -q tests/test_performance.py´
```

Desafios encontrados: 

- Acho que não entendi certas partes do sha-256 deterministico
###### - user_ids com exatos 13 caracteres seguem lógica diferente
###### - padrões específicos (ex: terminados em "_prime") têm regras especiais
- O calculo de influência demorou mais tempo que o esperado para fazer
- A detecção de anomalias foi complicada de ser feita