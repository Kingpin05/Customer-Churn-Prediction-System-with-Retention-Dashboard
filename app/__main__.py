"""Run the app locally: python -m app"""

from app import create_app

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8000, debug=False)
