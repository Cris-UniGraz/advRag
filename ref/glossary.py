from typing import Dict, List

def get_glossary(language: str) -> Dict[str, str]:
    """
    Devuelve un diccionario con un glosario de términos y sus explicaciones en el idioma especificado.
    
    Parámetros:
    language (str): El idioma para el que se debe obtener el glosario. Puede ser "german" o "english".
    
    Devuelve:
    Dict[str, str]: Un diccionario que contiene el glosario.
    """
    german_glossary = {
        "PSP-Code": "Eine eindeutige Kennung für eine Personalplanungsmaßnahme. Es handelt sich dabei um einen 8-stelligen Code (beginnend mit P - z.B. P0035123), welcher beispielsweise für den Personalaufnahmeprozess benötigt wird. Sie erhalten den PSP-Code beim Dekanat bzw. bei der planungsverantwortlichen Person Ihrer Einheit.",
        "flExam": "Ein Service von Lehr- und Studienservices, das von der uniIT technisch bereitgestellt wird (Eigenentwicklung der uniIT). Studierenden wird ermöglicht, im Rahmen festgelegter Prüfungswochen ihre Prüfungstermine (Slots) flexibel zu buchen für Lehrende wird dadurch der organisatorische Aufwand reduziert. Die Aufsicht wird zentral von Lehr- und Studienservices zur Verfügung gestellt.",
        "e-Auto": "Ein Elektroauto ist ein Fahrzeug, das von einem oder mehreren Elektromotoren angetrieben wird und seine Energie aus einer oder mehreren Batterien oder anderen Energiespeichern bezieht. Im Gegensatz zu herkömmlichen Fahrzeugen mit Verbrennungsmotor verwenden Elektroautos keine fossilen Brennstoffe wie Benzin oder Diesel und stoßen daher beim Fahren keine Abgase aus.",
        "e-Tankstelle": "Eine E-Tankstelle, oft auch als Elektroladestation oder Ladepunkt bezeichnet, ist ein Standort, an dem elektrische Energie zum Aufladen von Elektrofahrzeugen bereitgestellt wird. Im Kern handelt es sich dabei um eine Infrastruktureinrichtung, die es ermöglicht, die Akkumulatoren (Batterien) von Elektroautos, E-Bikes, elektrischen Motorrädern und anderen Fahrzeugen mit elektrischem Antrieb wieder aufzuladen.",
        "uniBOOK": "Ein Laptop, der von der Universität Graz für ihre MitarbeiterInnen zur Verfügung gestellt wird.",
        "Vortragenden-Account": "Lehrende an der Universität Graz haben ein zusätzliches Login auf von der uniIT betreuten Geräten in Hörsälen und Lehrräumen. Diese Vortragenden-Accounts sollen dazu dienen, die Sicherheit Ihrer persönlichen und der vertraulichen Daten Ihrer Einheit zu erhöhen. Melden Sie sich mit diesem eingeschränkten Account an Vortragenden-PCs in Hörsälen und Lehrräumen an, um im Falle eines missbräuchlichen Zugriffs (z.B. bei vergessenem Ausloggen oder bei Manipulationen am Gerät durch Dritte) vor Datendiebstahl geschützt zu sein.",
        "uniACCOUNT": "Lehrende an der Universität Graz haben ein zusätzliches Login auf von der uniIT betreuten Geräten in Hörsälen und Lehrräumen. Diese Vortragenden-Accounts sollen dazu dienen, die Sicherheit Ihrer persönlichen und der vertraulichen Daten Ihrer Einheit zu erhöhen. Melden Sie sich mit diesem eingeschränkten Account an Vortragenden-PCs in Hörsälen und Lehrräumen an, um im Falle eines missbräuchlichen Zugriffs (z.B. bei vergessenem Ausloggen oder bei Manipulationen am Gerät durch Dritte) vor Datendiebstahl geschützt zu sein.",
        "UNIGRAzcard": "Eine Bedienstetenkarte, die es Mitarbeitern, Studenten und Professoren ermöglicht, verschiedene Dienste der Universität Graz zu nutzen. Die Karte muss alle 30 Tage bei einem Update-Leser bzw. bei Online-Schlössern verlängert werden.",
        "unikat": "Das Suchportal, über das die meisten durch die UB zur Verfügung gestellten Medien: Bücher, E-Books, Zeitschriften, E-Journals, DVDs und vieles mehr gesucht (und gefunden!) werden können. unikat kann dabei auch externe Indizes durchsuchen, die Ihnen das Auffinden von (und, im Falle von entsprechenden Lizenzen, den Zugriff auf) mehr als 400 Millionen Zeitschriftenartikel aus etwa 150.000 Journals ebenso wie von e-Books aus allen Bereichen der Forschung ermöglichen.",
        "outro": "Beispiel."
    }
    
    english_glossary = {
        "PSP-Code": "A unique identifier for a personnel planning measure. It is an 8-digit code (starting with P - e.g. P0035123), which is required, for example, for the recruitment process. You can obtain the PSP-Code from the dean's office or the person responsible for planning in your unit.",
        "flExam": "A service of Teaching and Study Services, which is technically provided by uniIT (in-house development of uniIT). It enables students to book their exam dates (slots) flexibly within the framework of fixed exam weeks, thus reducing the organizational effort for lecturers. Supervision is provided centrally by Teaching and Student Services.",
        "e-Auto": "An electric car, is a vehicle that is powered by one or more electric motors and draws its energy from one or more batteries or other energy storage devices. Unlike conventional vehicles with an internal combustion engine, electric cars do not use fossil fuels such as petrol or diesel and therefore do not emit any exhaust fumes when driving.",
        "e-Tankstelle": "An electric filling station, often also referred to as an electric charging station or charging point, is a location where electrical energy is provided for charging electric vehicles. Essentially, it is an infrastructure facility that makes it possible to recharge the accumulators (batteries) of electric cars, e-bikes, electric motorcycles and other electrically powered vehicles.",
        "uniBOOK": "A laptop provided by the University of Graz for its employees.",
        "Vortragenden-Account": "Lecturers at the University of Graz have an additional login on devices managed by uniIT in lecture halls and classrooms. These lecturer accounts are intended to increase the security of your personal data and the confidential data of your unit. Log in to lecturer PCs in lecture halls and classrooms with this restricted account in order to be protected against data theft in the event of unauthorized access (e.g. if you forget to log out or if the device is tampered with by a third party).",
        "uniACCOUNT": "Lecturers at the University of Graz have an additional login on devices managed by uniIT in lecture halls and classrooms. These lecturer accounts are intended to increase the security of your personal data and the confidential data of your unit. Log in to lecturer PCs in lecture halls and classrooms with this restricted account to be protected against data theft in the event of unauthorized access (e.g. if you forget to log out or if the device is tampered with by a third party).",
        "UNIGRAzcard": "A staff card that enables employees, students and professors to use various services of the University of Graz. The card must be renewed every 30 days at an update reader or at online locks.",
        "unikat": "The search portal through which most of the media provided by the University Library can be searched (and found!): books, e-books, journals, e-journals, DVDs and much more. unikat can also search external indices, which enable you to find (and, if you have the appropriate licenses, access) more than 400 million journal articles from around 150,000 journals as well as e-books from all areas of research.", 
        "outro": "example"
    }
    
    if language == "german":
        return german_glossary
    elif language == "english":
        return english_glossary
    else:
        raise ValueError("Invalid language. Please use 'german' or 'english'.")
    
def find_glossary_terms_with_explanation(query: str, language: str) -> List[tuple[str, str]]:
    """
    Find glossary terms that appear in the query.
    Handles both single-word and multi-word terms.
    
    Args:
        query: The user query string
        glossary: Dictionary of terms and their explanations
    
    Returns:
        List of tuples containing matched terms and their explanations
    """
    glossary = get_glossary(language)

    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Find matching terms
    matches = []
    for term, explanation in glossary.items():
        # Convert term to lowercase for case-insensitive matching
        if term.lower() in query_lower:
            matches.append((term, explanation))
    
    # print(f"matches = {matches}")

    return matches


def find_glossary_terms(query: str, language: str) -> List[tuple[str, str]]:
    """
    Find glossary terms that appear in the query.
    Handles both single-word and multi-word terms.
    
    Args:
        query: The user query string
        glossary: Dictionary of terms and their explanations
    
    Returns:
        List of tuples containing matched terms and their explanations
    """
    glossary = get_glossary(language)

    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Find matching terms
    matches = []
    for term, explanation in glossary.items():
        # Convert term to lowercase for case-insensitive matching
        if term.lower() in query_lower:
            matches.append(term)
    
    # print(f"matches = {matches}")

    return matches