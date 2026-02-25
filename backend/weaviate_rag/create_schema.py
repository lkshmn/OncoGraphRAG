import weaviate
import logging
from dotenv import load_dotenv
from weaviate.classes.config import Property, DataType, Configure, ReferenceProperty

logger = logging.getLogger(__name__)

load_dotenv()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    client = weaviate.connect_to_local()

    try:
        # Check if we can connect
        client.get_meta()
    except Exception as e:
        logger.error(f"Could not connect to Weaviate: {e}. Is it running?")
        return

    try:
        client.collections.delete_all()
        logger.info("Deleted existing schema (Entities, Relations, Publications).")
    except Exception as e:
        logger.error(f"Error deleting schema: {e}")

    try:
        # 1. Create Entity Class
        client.collections.create(
            name="Entity",
            properties=[
                Property(name="name", data_type=DataType.TEXT),
                Property(name="type", data_type=DataType.TEXT), 
                # e.g., Disease, Treatment, Gene
            ],
            # Using basic text2vec-transformers (assumes container is running)
            # If no vectorizer is configured in your docker-compose, remove vectorizer_config
            vectorizer_config=Configure.Vectorizer.text2vec_transformers()
        )
        logger.info("Created class: Entity")

        # 2. Create Publication Class
        client.collections.create(
            name="Publication",
            properties=[
                Property(name="name", data_type=DataType.TEXT), # ID or Title
                Property(name="content", data_type=DataType.TEXT), # Abstract text
            ],
            vectorizer_config=Configure.Vectorizer.text2vec_transformers()
        )
        logger.info("Created class: Publication")

        # 3. Create Relation Class
        client.collections.create(
            name="Relation",
            properties=[
                Property(name="relation_predicate", data_type=DataType.TEXT), # e.g., treats, causes
            ],
            references=[
                ReferenceProperty(name="relation_subject", target_collection="Entity"),
                ReferenceProperty(name="relation_object", target_collection="Entity"),
                ReferenceProperty(name="hasPublication", target_collection="Publication")
            ],
            vectorizer_config=Configure.Vectorizer.text2vec_transformers()
        )
        logger.info("Created class: Relation")

    except Exception as e:
        logger.error(f"Error creating classes: {e}")

    try:
        final_schema = client.collections.list_all()
        # Simplified logging
        names = [k for k in final_schema.keys()]
        logger.info(f"Final schema classes in Weaviate: {names}")
    except Exception as e:
        logger.error(f"Error retrieving schema: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
