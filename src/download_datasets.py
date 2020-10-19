import urllib3

dataset_urls = [
    'http://cs.joensuu.fi/sipu/datasets/t4.8k.txt',
    'http://cs.joensuu.fi/sipu/datasets/s1.txt',
    'http://cs.joensuu.fi/sipu/datasets/a1.txt',
    'http://cs.joensuu.fi/sipu/datasets/birch1.txt',
    'http://cs.joensuu.fi/sipu/datasets/birch2.txt',
    'http://cs.joensuu.fi/sipu/datasets/birch3.txt',
    'http://cs.joensuu.fi/sipu/datasets/unbalance.txt',
    'http://cs.joensuu.fi/sipu/datasets/wine.txt',
    'http://cs.joensuu.fi/sipu/datasets/breast.txt',
    'http://cs.joensuu.fi/sipu/datasets/iris.txt',
    'http://cs.joensuu.fi/sipu/datasets/Aggregation.txt',
    'http://cs.joensuu.fi/sipu/datasets/dim064.txt',
]

if __name__ == '__main__':
    http = urllib3.PoolManager()
    for url in dataset_urls:
        response = http.request('GET', url)
        data = response.data.decode('UTF-8')
        name = url.split('/')[-1]
        with open(name, 'w') as file:
            file.write(data)
