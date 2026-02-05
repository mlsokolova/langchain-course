from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv("../.env")


def main():
    print("Hello from langchain-course!")
    information = """
 
Case, number and gender

Akkadian has two grammatical genders, masculine and feminine, with many feminine forms generated from masculine words by adding an -at suffix.

Formally, Akkadian has three numbers (singular, dual and plural) and three cases (nominative, accusative and genitive). However, the dual declension has disappeared from all but the noun paradigm; and even then, it is only used for things that naturally occur in pairs (most commonly body parts like eyes and ears). And in the plural numbers (including the dual), the accusative and genitive exhibit a single oblique case declension. Adjectives are declined exactly like nouns.

Akkadian, unlike Arabic, has mainly regular plurals (i.e. no broken plurals), although some masculine words take feminine plurals. In that respect, it is similar to Hebrew.

The nouns šarrum (king), šarratum (queen) and the adjective dannum (strong) will serve to illustrate the case system of Akkadian.

Noun and adjective paradigms
Noun (masc.)	Noun (fem.)	Adjective (masc.)	Adjective (fem.)
Nominative singular	šarr-um	šarr-at-um	dann-um	dann-at-um
Genitive singular	šarr-im	šarr-at-im	dann-im	dann-at-im
Accusative singular	šarr-am	šarr-at-am	dann-am	dann-at-am
Nominative dual	šarr-ān	šarr-at-ān	
Oblique* dual	šarr-īn	šarr-at-īn
Nominative plural	šarr-ū	šarr-āt-um	dann-ūt-um	dann-āt-um
Oblique plural	šarr-ī	šarr-āt-im	dann-ūt-im	dann-āt-im
* The oblique case includes the accusative and genitive.

As is clear from the above table, the adjective and noun endings differ only in the masculine plural. Certain nouns, primarily those referring to geography, can also form a locative ending in -um in the singular and the resulting forms serve as adverbials. These forms are generally not productive, but in the Neo-Babylonian the um-locative replaces several constructions with the preposition ina.

In the later stages of Akkadian the mimation (word-final -m) – along with nunation (dual final “-n”) – that occurs at the end of most case endings has disappeared, except in the locative. Later, the nominative and accusative singular of masculine nouns collapse to -u and in Neo-Babylonian most word-final short vowels are dropped. As a result case differentiation disappeared from all forms except masculine plural nouns. However many texts continued the practice of writing the case endings, but not consequently and often incorrectly. Also, the most important contact language was Aramaic which also lacked case differentiation, so it’s possible that the loss of cases differentiation in Akkadian was not only due to phonological phenomena.
"""
    summary_template = f"""
    given text  {information} describes some phenomena. I want a short summary of this text
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
