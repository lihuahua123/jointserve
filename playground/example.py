from preble.server.server import start_server_and_load_models
start_server_and_load_models(
    devices=[0],
    host='0.0.0.0',
    port=8001,
    model_name="/hy-tmp/"
)