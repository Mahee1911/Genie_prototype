import dotenv
dotenv.load_dotenv()

if __name__ == "__main__":
    from core.app import app
    from route import upload

    app.run(host='0.0.0.0', port=8080, debug=True)
