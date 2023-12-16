# I stole this code shamelessly from: https://github.com/samiriff/mars-ode-data-access
# Their license:
#
# The MIT License (MIT)
# Copyright (c) 2018 Massachusetts Institute of Technology
#
# Author: Guillaume Rongier
# This software has been created in projects supported by the US National
# Science Foundation and NASA (PI: Pankratius)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from six.moves.urllib.request import urlopen
from xml.dom import minidom
from collections import OrderedDict
import re


class Query:
    def __init__(self) -> None:
        pass

    def query(self, query: dict) -> list:
        links = self.query_files_urls(query=query)
        return links

    @staticmethod
    def get_query_url(query: dict):
        ODE_REST_base_url = "http://oderest.rsl.wustl.edu/live2/?"

        target = query["target"]
        mission = query["mission"]
        instrument = query["instrument"]
        product_type = query["product_type"]
        western_lon = query["western_lon"]
        eastern_lon = query["eastern_lon"]
        min_lat = query["min_lat"]
        max_lat = query["max_lat"]
        min_ob_time = query["min_ob_time"]
        max_ob_time = query["max_ob_time"]
        product_id = query["product_id"]
        query_type = query["query_type"]
        output = query["output"]
        results = query["results"]
        number_product_limit = query["number_product_limit"]
        result_offset_number = query["result_offset_number"]

        target = "target=" + target
        mission = "&ihid=" + mission
        instrument = "&iid=" + instrument
        product_type = "&pt=" + product_type
        if western_lon is not None:
            western_lon = "&westernlon=" + str(western_lon)
        else:
            western_lon = ""
        if eastern_lon is not None:
            eastern_lon = "&easternlon=" + str(eastern_lon)
        else:
            eastern_lon = ""
        if min_lat is not None:
            min_lat = "&minlat=" + str(min_lat)
        else:
            min_lat = ""
        if max_lat is not None:
            max_lat = "&maxlat=" + str(max_lat)
        else:
            max_lat = ""
        if min_ob_time != "":
            min_ob_time = "&mincreationtime=" + min_ob_time
        if max_ob_time != "":
            max_ob_time = "&maxcreationtime=" + max_ob_time
        if product_id != "":
            product_id = "&productid=" + product_id

        if query_type != "":
            query_type = "&query=" + query_type
        if results != "":
            results = "&results=" + results
        if output != "":
            output = "&output=" + output
        if number_product_limit != "":
            number_product_limit = "&limit=" + str(number_product_limit)
        if result_offset_number != "":
            result_offset_number = "&offset=" + str(result_offset_number)

        return (
            ODE_REST_base_url
            + target
            + mission
            + instrument
            + product_type
            + western_lon
            + eastern_lon
            + min_lat
            + max_lat
            + min_ob_time
            + max_ob_time
            + query_type
            + results
            + output
            + number_product_limit
            + result_offset_number
            + product_id
        )

    @staticmethod
    def get_files_urls(query_url, file_name="*", print_info=False) -> list:
        final_file_urls = []
        url = urlopen(query_url)
        query_results = url.read()
        xml_results = minidom.parseString(query_results)
        url.close()

        error = xml_results.getElementsByTagName("Error")
        if len(error) > 0:
            print("\nError:", error[0].firstChild.data)
            return []

        limit_file_types = "Product"
        file_name = file_name.replace("*", ".")

        products = xml_results.getElementsByTagName("Product")
        file_urls = OrderedDict()

        for product in products:
            product_files = product.getElementsByTagName("Product_file")
            product_id = product.getElementsByTagName("pdsid")[0]

            if print_info == True:
                print("\nProduct ID:", product_id.firstChild.data)

            for product_file in product_files:
                file_type = product_file.getElementsByTagName("Type")[0]
                file_url = product_file.getElementsByTagName("URL")[0]
                file_description = product_file.getElementsByTagName("Description")[0]
                local_filename = file_url.firstChild.data.split("/")[-1]
                local_file_extension = local_filename.split(".")[-1]

                if re.search(file_name, local_filename) is not None:
                    # Restriction on the file type to download
                    if len(limit_file_types) > 0:
                        # If match, get the URL
                        if file_type.firstChild.data == limit_file_types:
                            file_urls[file_url.firstChild.data] = (
                                product_id.firstChild.data,
                                file_description.firstChild.data,
                            )
                            # i guess ill just add it here for now
                            final_file_urls.append(file_url.firstChild.data)

                            if print_info == True:
                                # print(
                                #     "File name:",
                                #     file_url.firstChild.data.split("/")[-1],
                                # )
                                # print("Folder name:", file_url.firstChild.data)
                                # print("Description:", file_description.firstChild.data)
                                pass

                    # No restriction on the file type to download
                    else:
                        file_urls[file_url.firstChild.data] = (
                            product_id.firstChild.data,
                            file_description.firstChild.data,
                        )

                        if print_info == True:
                            # print("File name:", file_url.firstChild.data.split("/")[-1])
                            # print("Description:", file_description.firstChild.data)
                            pass

        return final_file_urls

    def query_files_urls(self, query: dict) -> list:
        # Returns a list of products with selected product metadata that meet the query parameters
        query_type = "product"
        # Controls the return format for product queries or error messages
        output = "XML"
        # For each product found return the product files and IDS
        results = "fp"
        query["output"] = output
        query["results"] = results
        query["query_type"] = query_type

        query_url = self.get_query_url(query)

        # print("Query URL:", query_url)
        # print("\nFiles that will be downloaded (if not previously downloaded):")
        file_urls = self.get_files_urls(query_url, query["file_name"], print_info=True)
        return file_urls


QUERY_DICT = {
    "target": "mars",
    "mission": "MRO",
    "instrument": "CTX",
    "product_type": "EDR",
    "western_lon": "",
    "eastern_lon": "",
    "min_lat": "",
    "max_lat": "",
    "min_ob_time": "",
    "max_ob_time": "",
    "product_id": "P13_00621*",
    "query_type": "",
    "results": "",
    "number_product_limit": "10",
    "result_offset_number": "",
    "file_name": "*.IMG",
}

if __name__ == "__main__":
    Q = Query()
    links = Q.query(QUERY_DICT)
    print(links)
