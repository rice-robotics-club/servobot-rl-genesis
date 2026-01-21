# Configuration Files

In contrast to the previous messy strategy of creating configuration files with
poorly defined options/fields, we define schemas in the `schemas` folder, which define the object
layout of our training configuration files. This uses the JSONSchema standard. If using an editor
that provides language server support for YAML, intellisense/code completion will be provided based
on the schema. This will check the config files to ensure type compliance with the expected config
file layout. In order for this intellisense to work, the following comment should be placed
at the top of the config file:

```yaml
# yaml-language-server: $schema=./schemas/config.json
```

This will inform the YAML language server of the schema to validate against.

Each conceptual component should have its own component in the training file. For example,
the "runner" component of training, which defines the rsl_rl runner configuration, can be
configured under the runner section of a config file:

```yaml
runner:
  class_name: OnPolicyRunner
  num_steps_per_env: 24
  max_iterations: 1500
  ...
```

This is compliant with the default config file layout provided by rsl_rl (although poorly documented),
exemplified by [example.yaml](example.yaml). It is important to note that the provided configuration layout
has fields dependent on the value of `class_name`.

## Extending

To add more configuration options, define an additional JSONSchema in `schemas`, and add it as a property 
of [`schemas/config.json`](schemas/config.json). It is recommended to use an LLM to produce this. This will 
update the schema, and you can then add your options under the name you have chosen:

```json
// example.json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "example.json",
  "title": "ExampleConfig",
  "type": "object",
  "properties": {
    "example_option1": {
      "type": "string"
    }
  },
  "required": ["example_option1"],
  "additionalProperties": false
}

// config.json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "config.json",
  "title": "Config",
  "type": "object",
  "properties": {
    ...
    "example": {
      "$ref": "example.json"
    }
  },
  "required": ["example"],
  "additionalProperties": false
}
```

```yaml
runner:
  class_name: OnPolicyRunner
  num_steps_per_env: 24
  max_iterations: 1500
  ...
example:
  example_option1: "wow"
  ...
```
