from pymilvus import connections, utility
# from langchain_milvus import Milvus

# Conectar a Milvus
connections.connect()

# Obtener todas las colecciones
all_collections = utility.list_collections()

# Colecciones a mantener
keep_collections = ["uni_test_2_0_alles", "uni_test_2_0_alles_hyde", "uni_test_2_0_alles_children", "uni_test_3_0_de", "uni_test_3_0_en"]

# Iterar sobre todas las colecciones y borrar las que no están en la lista de exclusión
for collection_name in all_collections:
    if collection_name not in keep_collections:
        utility.drop_collection(collection_name)
        print(f"Colección '{collection_name}' borrada.")
    else:
        print(f"Colección '{collection_name}' mantenida.")

# Desconectar de Milvus
connections.disconnect("default")

print("Proceso completado.")
