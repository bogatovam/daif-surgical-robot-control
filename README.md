# daif-surgical-robot-control

# Воспроизведение результатов
### MuJoCo

В папке notebook для каждого агента и задачи содержаться соответствующие блокноты. Достаточно запустить нужный  в среде Google Colab. Симулятор работает исключительно на Linux. 
**Важно:** после выполнения первого блока кода с установкой симулятора необходимо перезапустить среду выполнения.

Задача FetchReach решается за ~полчаса.
Задача FetchPickAndPlace решается за ~12 часов.

### SurRol

Для запуска агента на этом симуляторе рекомендуется использовать локальную машину, так как обучение этих агентов требует значительно больше вермени.
Для начала необходимо установить кастомную версию симулятора. См. https://github.com/bogatovam/surrol/tree/aif-dev

Далее, в созданной виртуальной среде с симулятором, можно запускать агента. 
Для конфигурации см. https://github.com/bogatovam/daif-surgical-robot-control/tree/main/config/resources
Для запуска достаточно открыть файл https://github.com/bogatovam/daif-surgical-robot-control/blob/main/scripts/mdp_daif_agent.py и вызвать метод:

```
train_agent_according_config(get_config(env_id='NeedlePickViaGrasp-v0', device='cpu'))
```

NeedleReach ~32 часа
NeedleGrasp ~96 часов
