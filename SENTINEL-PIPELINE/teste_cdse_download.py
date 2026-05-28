"""
Teste de autenticação e download no Copernicus Data Space Ecosystem (CDSE).

Objetivo:
1. Testar se o login do config.py gera token corretamente.
2. Testar se conseguimos acessar metadados de um produto Sentinel-2.
3. Testar se conseguimos baixar o produto pelo endpoint /$value.
4. Descobrir se o erro 403 é problema de código ou de permissão da conta.

Uso:
    python3 teste_cdse_download.py
"""

import requests

from config import (
    CDSE_TOKEN_URL,
    CDSE_USER,
    CDSE_PASSWORD,
)


# Produto Sentinel-2 que apareceu no seu log
PRODUTO_ID = "26df8eca-34d0-4de7-a7ec-a64ac8e633d2"

ODATA_SEARCH = "https://catalogue.dataspace.copernicus.eu/odata/v1"
ODATA_DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1"


def obter_token():
    print("Gerando token CDSE...")

    resp = requests.post(
        CDSE_TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": CDSE_USER,
            "password": CDSE_PASSWORD,
            "grant_type": "password",
        },
        timeout=30,
    )

    print("Status token:", resp.status_code)

    if resp.status_code != 200:
        print("Erro ao gerar token:")
        print(resp.text[:1000])
        raise SystemExit(1)

    token = resp.json()["access_token"]

    print("Token gerado com sucesso.")
    print("Tamanho do token:", len(token))

    return token


def testar_metadados_catalogo(token):
    print("\nTestando metadados pelo catálogo...")

    url = f"{ODATA_SEARCH}/Products({PRODUTO_ID})"

    resp = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
        },
        timeout=30,
    )

    print("Status metadados catálogo:", resp.status_code)

    if resp.status_code != 200:
        print("Erro:")
        print(resp.text[:1000])
        return

    data = resp.json()

    print("Nome:", data.get("Name"))
    print("Online:", data.get("Online"))
    print("ContentLength:", data.get("ContentLength"))
    print("ContentDate:", data.get("ContentDate"))


def testar_nodes_download(token):
    print("\nTestando Nodes pelo domínio de download...")

    url = f"{ODATA_DOWNLOAD}/Products({PRODUTO_ID})/Nodes"

    resp = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
        },
        timeout=30,
    )

    print("Status Nodes:", resp.status_code)

    if resp.status_code != 200:
        print("Erro:")
        print(resp.text[:1000])
        return

    data = resp.json()

    itens = data.get("result", data.get("value", []))

    print("Quantidade de nodes raiz:", len(itens))

    for item in itens[:5]:
        print("- Id:", item.get("Id"))
        print("  Name:", item.get("Name"))
        print("  ChildrenNumber:", item.get("ChildrenNumber"))
        print("  Nodes URI:", item.get("Nodes", {}).get("uri"))


def testar_download_produto(token):
    print("\nTestando download do produto inteiro via /$value...")

    url = f"{ODATA_DOWNLOAD}/Products({PRODUTO_ID})/$value"

    print("URL:")
    print(url)

    resp = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Range": "bytes=0-1023",
        },
        stream=True,
        allow_redirects=True,
        timeout=60,
    )

    print("Status download:", resp.status_code)
    print("URL final:", resp.url)
    print("Content-Type:", resp.headers.get("Content-Type"))
    print("Content-Length:", resp.headers.get("Content-Length"))
    print("Accept-Ranges:", resp.headers.get("Accept-Ranges"))

    if resp.status_code not in (200, 206):
        print("\nErro no download:")
        print(resp.text[:1000])
        return False

    try:
        chunk = next(resp.iter_content(chunk_size=128))
        print("Primeiros bytes:", chunk[:80])
    except Exception as exc:
        print("Erro lendo primeiros bytes:", exc)
        return False

    return True


def main():
    token = obter_token()

    testar_metadados_catalogo(token)
    testar_nodes_download(token)

    ok_download = testar_download_produto(token)

    print("\nResumo:")

    if ok_download:
        print("✅ A conta consegue baixar o produto pelo /$value.")
        print("Próximo passo: adaptar o pipeline para baixar o produto completo e extrair as bandas localmente.")
    else:
        print("❌ A conta NÃO conseguiu baixar o produto pelo /$value.")
        print("Provável causa: permissão/credencial da conta CDSE ou bloqueio do método de download.")
        print("Próximo passo: testar com uma conta própria do Copernicus Data Space no config.py.")


if __name__ == "__main__":
    main()