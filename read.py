from warcio import ArchiveIterator
import os

def cc_wet_warc(path: str, lang: str = None):
    """ 
    Reads CommonCrawl files in WET format and processes them into a list of documents.
    Allows a lang filtering argument.
    """

    corpus = []

    with open(path, 'rb') as stream:
        for i, record in enumerate(ArchiveIterator(stream)):
            try:
                # There is a record in each WARC file with info about the extraction
                if record.rec_type == "warcinfo":
                    #warcinfo = record.content_stream().read()
                    continue
                # Check if record contains header
                if record.rec_headers:
                    # Get metadata
                    header = record.rec_headers
                    uri = str(header['uri']) if 'uri' in header else str(header['WARC-Target-URI'])
                    lang_list = header['WARC-Identified-Content-Language'].split(",") if 'WARC-Identified-Content-Language' in header else None
                    # Get plain text
                    text = record.content_stream().read().decode()
                    # Create Document object
                    current_document = {
                        "id": i,
                        "uri": uri,
                        "lang": lang_list,
                        "source": "cc",
                        "text": text
                    }

                    if lang_list != None:
                        # Append if webpage only has one specified language
                        if lang and lang_list == [lang]:
                            corpus.append(current_document)
                        # Append if language is not specified
                        if not lang:
                            corpus.append(current_document)
                        else:
                            continue
            except:
                pass

    return corpus

def wikipedia(path: str):
    corpus = []

    for i, doc in enumerate(os.listdir(path)):
        with open(path+doc, "r") as fin:
            # Create Document object
            current_document = {
                "id": i,
                "uri": "",
                "lang": "",
                "source": "wiki",
                "text": fin.read()
            }

        corpus.append(current_document)

    return corpus