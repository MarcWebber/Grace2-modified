# Grace2-modified

## requirements

```
python >= 3.9
java >= 1.8
openclover
maven >= 3.8.5
```

## APIS:

```java
void js_extrat(String js_part_path)
void mc_extract(String path)
```

These two apis are for analyzing  open clover report

```java
void OCP()
```

OCP is for generating priority ranks with additional greedy

```java
void model_predict()
```

This function use model to predict the priority ranks with more features

## USAGE

```bash
java -jar TCP-0.0.1-SNAPSHOT.jar <YOUR_CODE'S_ROOT_PATH>
```

after this, you can see the result on the console