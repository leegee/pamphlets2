import lancedb

from lib.corpus_config import LANCE_INDEXES_DIR

db = lancedb.connect(str(LANCE_INDEXES_DIR))

table = db.open_table("local__macberth__1650_1699")

print("schema:")
print(table.schema)

print()
print("indices:")
for index in table.list_indices():
    print(index)
    index = next(
        index
        for index in table.list_indices()
        if index.name == "vector_idx"
    )

    print()
    print("vector index:")
    print(index.index_details)