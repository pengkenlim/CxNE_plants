import ast

def parse_parameters(filename):
    """Parses a text file containing Python variables and returns a dictionary.

    Args:
        filename (str): The path to the text file.

    Returns:
        dict: A dictionary containing the parsed variables.

    Raises:
        SyntaxError: If the text file contains invalid Python syntax.
    """

    parameters = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip empty lines and comments

            try:
                key, value = line.split('=')
                key = key.strip()
                value = ast.literal_eval(value.strip())
                parameters[key] = value
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing line '{line}': {e}")

    return parameters

# Example usage
if __name__ == "__main__":
    parameters = parse_parameters("parameters.txt")
    print(parameters)