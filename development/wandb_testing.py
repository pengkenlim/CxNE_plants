# %%
import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="CxNE_Viridiplantae",

    name= "Test_2",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 100
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"epoch": epoch,
               "losses":{ "training_mode": {"SEMBP": {"taxid3702": {"tr_loss": loss, "tst_loss": loss},
                                                      "taxid4081": {"tr_loss": loss, "tst_loss": loss}}},
                                            "ASEMBP": {"tr_loss": loss, "tst_loss": loss}
                                            },
                            "inference_mode": { "SFBP":{ "taxid3702": {"tr_loss": loss, "tst_loss": loss},
                                                        "taxid4081": {"tr_loss": loss, "tst_loss": loss}},
                                                "ASFBP":{"tr_loss": loss, "tst_loss": loss}
                                                },
                            "training_order": {"taxid3702": 1, "taxid4081": 0}
                   })

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
# %%
