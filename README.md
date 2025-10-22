# Usage guide

1. Use a compatible python version (3.9 mostly).
2. `python3 -m venv ./venv`.
3. `source ./venv/bin/activate`.
4. `pip3 install rasa`.
5. `rasa init` -> creates the project structure. Specify a folder in which the files are generated.
6. Delete all the generated files in the folder and copy all the files from here.
7. Start the `rasa/duckling` docker container to start the duckling server and make its entry in `endpoints.yml`, if needed. Look at [Duckling](https://github.com/facebook/duckling "Ducling") to learn more about its capabilities. (Optional)

## Make your changes. To test, run the following

### Start the rasa action server

- Add the `OPENAI_API_KEY` to `.env`
- In a terminal (activate the virtual env first), then `rasa train` and then `rasa run actions` to start the actions server on port 5055.

### Test the program in a shell

- In another terminal (activate the virtual env first), run `rasa shell`. The terminal starts up and you can interact with the agent. Uses port 5005

### To make it available as an API, run the following

- In a terminal (activate the virtual env first), then `rasa train` and then `rasa run actions` to start the actions server on port 5055.
- In another terminal (activate the virtual env first), start the main rasa server. Run `rasa run --enable-api --cors "*"` to start it. Replace "*" with the domain which should interact with it. Uses port 5005. Specify a custom port using `-p` flag. To use authorization, pass ` --auth-token "my_secret_token"` to the above command. Then, in the request send the `Authorization: Bearer my_secret_token` header.

That's pretty much it. Any modifications to the code, retrain and restart the servers. See the [Sample Questions](Sample_Questions.txt) file to see some sample questions that can be asked.
