# curl -X POST "http://localhost:3333/extract_topics/" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "possible_topics": ["credit", "food", "debt", "bank"],
#     "messages": [
#       {"sender": "user", "message": "Добрий день, які у вас умови кредиту?"},
#       {"sender": "operator", "message": "Добрий день, раді вітати!"},
#       {"sender": "user", "message": "Чи можливо розстрочення платежів?"},
#       {"sender": "operator", "message": "Так, ми можемо обговорити розстрочення."}
#     ]
#   }'

# curl -X POST "http://localhost:3333/extract_topics/" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "possible_topics": ["credit", "food", "debt", "bank"],
#     "messages": [
#       {"sender": "user", "message": "Я хотів би обговорити можливість реструктуризації мого кредиту."},
#       {"sender": "operator", "message": "Звичайно, ми готові обговорити нові умови."},
#       {"sender": "user", "message": "Чи можливо переглянути графік погашення боргу?"},
#       {"sender": "operator", "message": "Так, ми можемо зробити це для вас."}
#     ]
#   }'

# curl -X POST "http://localhost:3333/extract_topics/" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "possible_topics": ["credit", "food", "debt", "bank"],
#     "messages": [
#       {"sender": "user", "message": "Які варіанти погашення боргу у вас є?"},
#       {"sender": "operator", "message": "Ми можемо запропонувати різні способи погашення."},
#       {"sender": "user", "message": "Чи можливо розстрочити платежі за боргом?"},
#       {"sender": "operator", "message": "Так, ми можемо розглянути такий варіант."}
#     ]
#   }'

curl -X POST "http://localhost:3333/extract_topics/" \
  -H "Content-Type: application/json" \
  -d '{
    "possible_topics": ["зацікавлений", "не зацікавлений", "незручно говорити"],
    "messages": [ 
      {"sender": "user", "message": "ку"},
      {"sender": "user", "message": "Привіт! Чи цікавить вам покупка авто?"},
      {"sender": "user", "message": "я на роботі, передзвоніть пізніше"}
    ]
  }'

