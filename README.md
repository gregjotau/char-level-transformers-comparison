# Comparing Transformer variants on character-level transduction tasks

This repository contains the code for the paper "Comparing Transformer variants for character-level transduction tasks".

## How training was done on Educloud

```
ssh ec-gregorjt@fox.educloud.no
```

`~/.ssh/config`, use "Control" options to persist connection, no need to type one-time code+pw too many times. 

```
Host edu
  HostName fox.educloud.no
  User ec-gregorjt
  IdentityFile ~/.ssh/id_rsa
  ControlMaster auto
  ControlPath ~/.ssh/controlsock-%r@%h:%p
  ControlPersist yes 
```


