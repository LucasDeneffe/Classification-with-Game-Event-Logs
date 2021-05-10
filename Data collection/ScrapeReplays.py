import requests

"""url = r'https://lotv.spawningtool.com/zip/?patch=&before_time=&tag=13&after_played_on=&coop=&before_played_on=&after_time=&query=&order_by=&p=1'
r = requests.get(url, allow_redirects=True)

with open ('zip_replays.zip', 'wb') as f:
    f.write(r.content)"""

# ZvZ
for i in range(1, 265):
    url = r"https://lotv.spawningtool.com/zip/?before_time=&coop=&before_played_on=&query=&tag=13&after_time=&patch=&after_played_on=&order_by=&p={}".format(
        i
    )
    r = requests.get(url, allow_redirects=True)

    with open(f"Data collection/Replays/zip_replays_ZvZ_{i}.zip", "wb") as f:
        f.write(r.content)
    if i % 20 == 0:
        print(f"page {i} is done")

# PvP
for i in range(1, 191):
    url = r"https://lotv.spawningtool.com/zip/?before_played_on=&after_played_on=&patch=&coop=&after_time=&before_time=&query=&order_by=&adv=1&tag=9&p={}".format(
        i
    )
    r = requests.get(url, allow_redirects=True)

    with open(f"Data collection/Replays/zip_replays_PvP_{i}.zip", "wb") as f:
        f.write(r.content)
    if i % 20 == 0:
        print(f"page {i} is done")

# PvT
for i in range(1, 438):
    url = r"https://lotv.spawningtool.com/zip/?before_time=&tag=10&after_played_on=&after_time=&coop=&patch=&query=&before_played_on=&order_by=&p={}".format(
        i
    )
    r = requests.get(url, allow_redirects=True)

    with open(f"Data collection/Replays/zip_replays_PvT_{i}.zip", "wb") as f:
        f.write(r.content)
    if i % 20 == 0:
        print(f"page {i} is done")

# PvZ
for i in range(1, 493):
    url = r"https://lotv.spawningtool.com/zip/?tag=11&patch=&query=&after_time=&before_time=&coop=&order_by=&after_played_on=&before_played_on=&p={}".format(
        i
    )

    r = requests.get(url, allow_redirects=True)

    with open(f"Data collection/Replays/zip_replays_PvZ_{i}.zip", "wb") as f:
        f.write(r.content)
    if i % 20 == 0:
        print(f"page {i} is done")

# TvZ


for i in range(1, 66):
    url = r"https://lotv.spawningtool.com/zip/?tag=3&before_time=&after_played_on=&order_by=&before_played_on=&after_time=&coop=&query=&patch=&p={}".format(
        i
    )

    r = requests.get(url, allow_redirects=True)

    with open(f"Data collection/Replays/zip_replays_TvZ_{i}.zip", "wb") as f:
        f.write(r.content)
    if i % 20 == 0:
        print(f"page {i} is done")

# TvT
for i in range(1, 203):
    url = r"https://lotv.spawningtool.com/zip/?tag=12&after_time=&after_played_on=&query=&patch=&before_time=&coop=&before_played_on=&order_by=&p={}".format(
        i
    )

    r = requests.get(url, allow_redirects=True)

    with open(f"Data collection/Replays/zip_replays_TvT_{i}.zip", "wb") as f:
        f.write(r.content)
    if i % 20 == 0:
        print(f"page {i} is done")
