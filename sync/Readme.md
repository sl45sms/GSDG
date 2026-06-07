here i will put some scripts to sync outputs to argila.glossapi.gr

so to take external ip you can use this command:

```bash
curl ifconfig.me
```

copy outputs folder
```
scp -r -p -v /users/p-skarvelis/GSDG/outputs argilla:/mnt/data/GSDG_OUTPUTS/
```


