#include <fstream> 
#include <iostream>

#include <nlohmann/json-schema.hpp>

using nlohmann::json;
using nlohmann::json_schema::json_validator;

int main(int argc, char *argv[]) {
    json schema_json, example_json;

    std::ifstream schema_file("schema.json");
    if (!schema_file.is_open()) {
        std::cerr  << "Error opening schema file.\n";
    }
    schema_file >> schema_json;

    json_validator validator;
    validator.set_root_schema(schema_json);

    std::ifstream data_file("example.json");
    if (!data_file.is_open()) {
        std::cerr << "Error opening data file.\n";
        return 1;
    }
    data_file >> example_json;

    try {
        validator.validate(example_json);
        std::cout << "JSON is valid.\n";
    } catch (const std::exception &e) {
        std::cerr << "Validation error: " << e.what() << "\n";
    }

    return EXIT_SUCCESS;
}
