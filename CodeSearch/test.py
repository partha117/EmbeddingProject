import jsonlines

jfile = jsonlines.open("java_test_0.jsonl","r")
count = 0
for item in jfile:
    count += 1

print(count)