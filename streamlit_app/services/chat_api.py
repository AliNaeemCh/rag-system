import json
import requests

from app.core.config import settings



def stream_chat(message, session_id, mode):

    payload = {
        "message": message,
        "session_id": session_id,
        "mode": mode
    }


    headers = {
        "X-API-Key": settings.API_KEY
    }


    with requests.post(
        settings.CHAT_URL,
        json=payload,
        headers=headers,
        stream=True,
        timeout=300
    ) as res:

        if res.status_code >= 400:
            raise Exception(
                "Error from backend"
            )


        for line in res.iter_lines(
            decode_unicode=True
        ):

            if not line:
                continue


            if line.startswith("data: "):

                yield json.loads(
                    line.replace(
                        "data: ",
                        ""
                    )
                )