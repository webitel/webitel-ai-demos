import asyncio
import aiohttp

async def fetch(session, url, data):
    async with session.post(url, json=data) as response:
        return await response.json()

async def main():
    english_text = "Hello world!"
    url = "http://127.0.0.1:8000/"
    tasks = []

    # Number of requests you want to make
    num_requests = 1000

    async with aiohttp.ClientSession() as session:
        for _ in range(num_requests):
            task = asyncio.create_task(fetch(session, url, english_text))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        for i, response_body in enumerate(responses):
            print(f"Response {i+1}: {response_body['embedding']}")

if __name__ == "__main__":
    asyncio.run(main())
