from src.data.cases import discover_cases


cases = discover_cases(
    images_dir="data/multicase/images",
    labels_dir="data/multicase/labels"
)


print("Number of discovered cases:")
print(len(cases))

print("First case:")
print(cases[0])

print("Last case:")
print(cases[-1])


assert len(cases) == 10

assert cases[0]["id"] == "liver_0"

assert cases[-1]["id"] == "liver_9"